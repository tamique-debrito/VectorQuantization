import random
import copy
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple, Any

from Evolutionary.base_unit import base_unit_factory
from Evolutionary.abstract_unit import Mat
from Evolutionary.train_unit import TrainUnit, run_train_session, simple_eval
from Evolutionary.mutations import simple_mutate, splice_units


# The 5 lookup tables a BaseUnit holds. Order is meaningful only for serialization.
_TABLE_NAMES = (
    "FORWARD_TABLE",
    "FIND_BACK_INFO_TABLE",
    "BACKWARD_TABLE",
    "FIND_UPDATE_TABLE",
    "APPLY_UPDATE_TABLE",
)


def _serialize_unit(unit) -> dict:
    """Extract just the data needed to reconstruct unit behavior in a worker.

    Per-instance attributes (set by mutate / splice / construct) shadow class
    attributes; getattr resolves either way, so this works for both random
    factory-built units and mutated descendants.
    """
    return {
        "w_idx": unit.w.w_idx,
        "tables": {name: getattr(unit, name) for name in _TABLE_NAMES},
    }


def _evaluate_unit_worker(payload):
    """Worker entry point. Reconstructs a BaseUnit from serialized data, runs
    the same fitness eval the main process would have run, returns the score.

    The dynamically-created `BaseUnit` class can't pickle across processes, so
    we ship only `(tables, w_idx)` and rebuild the class locally inside the
    worker. The freshly-built class has random class-level tables, which we
    override with the unit's tables via instance attributes — same trick as
    `simple_mutate` and `construct_for_function`.
    """
    serialized, eval_kwargs = payload
    BaseUnit = base_unit_factory(eval_kwargs["num_data_obj"], num_mat=eval_kwargs["num_op_obj"])
    unit = BaseUnit(Mat(serialized["w_idx"]))
    for name, table in serialized["tables"].items():
        setattr(unit, name, table)
    return TrainUnit.static_evaluate_training_performance(
        unit,
        eval_kwargs["trains_per_unit"],
        simple_eval,
        eval_kwargs["num_examples"],
        eval_kwargs["num_epochs"],
        eval_kwargs["num_data_obj"],
        eval_kwargs["dataset_type"],
    )

class SimpleEvolutionaryAlg:
    def __init__(self,
                 pop_size: int,
                 num_data_obj: int,
                 trains_per_unit: int,
                 examples_per_train: int,
                 epochs_per_train: int,
                 mutation_probability: float,
                 top_q_percent: float = 0.7,
                 num_op_obj: Optional[int] = None,
                 dataset_type: str = "random",
                 initial_population: Optional[List[Any]] = None,
                 workers: int = 1,
                ):
        # If an initial_population is supplied, pop_size is taken from it.
        if initial_population is not None:
            pop_size = len(initial_population)
        self.pop_size = pop_size
        self.num_data_obj = num_data_obj
        self.num_op_obj = num_op_obj
        self.trains_per_unit = trains_per_unit
        self.examples_per_train = examples_per_train
        self.epochs_per_train = epochs_per_train
        self.mutation_probability = mutation_probability
        self.top_q_percent = top_q_percent
        self.dataset_type = dataset_type
        # workers > 1 enables intra-run parallelism: per-unit fitness evaluations
        # are dispatched to a process pool. Worth it once population eval dominates
        # mutation / splice / sort costs (i.e. pop_size in the high hundreds and up).
        self.workers = workers
        self.population: List[Tuple[Any, float]] = [] # List of (BaseUnit, performance) tuples
        # Create a TrainUnit instance to get num_examples and num_epochs for static calls
        self.train_config = TrainUnit(base_unit_factory(self.num_data_obj), self.examples_per_train, self.epochs_per_train, self.dataset_type)
        if initial_population is not None:
            self._initialize_population_from_seed(initial_population)
        else:
            self._initialize_population()
        print(f"Algorithm initialized with parameters: pop_size={self.pop_size}, num_data_obj={self.num_data_obj}, num_op_obj={self.num_op_obj}, trains_per_unit={self.trains_per_unit}, examples_per_train={self.examples_per_train}, epochs_per_train={self.epochs_per_train}, mutation_probability={self.mutation_probability}, top_q_percent={self.top_q_percent}")

    def set_train_parameters(self,
                 trains_per_unit: Optional[int] = None,
                 examples_per_train: Optional[int] = None,
                 epochs_per_train: Optional[int] = None,
                 mutation_probability: Optional[float] = None,
                 top_q_percent: Optional[float] = None):
        print(f"Updating training parameters: trains_per_unit={trains_per_unit}, examples_per_train={examples_per_train}, epochs_per_train={epochs_per_train}, mutation_probability={mutation_probability}, top_q_percent={top_q_percent}")
        if trains_per_unit is not None:
            self.trains_per_unit = trains_per_unit
        if examples_per_train is not None:
            self.examples_per_train = examples_per_train
        if epochs_per_train is not None:
            self.epochs_per_train = epochs_per_train
        if mutation_probability is not None:
            self.mutation_probability = mutation_probability
        if top_q_percent is not None:
            self.top_q_percent = top_q_percent
        
        # Re-initialize train_config with potentially updated parameters
        self.train_config = TrainUnit(base_unit_factory(self.num_data_obj), self.examples_per_train, self.epochs_per_train, self.dataset_type)

    def _eval_kwargs(self, trains_per_unit: int) -> dict:
        """Common kwargs needed to reconstruct + evaluate a unit in a worker."""
        return {
            "num_data_obj": self.num_data_obj,
            "num_op_obj": self.num_op_obj,
            "trains_per_unit": trains_per_unit,
            "num_examples": self.examples_per_train,
            "num_epochs": self.epochs_per_train,
            "dataset_type": self.dataset_type,
        }

    def _evaluate_units(self, units: List[Any], trains_per_unit: Optional[int] = None) -> List[float]:
        """Evaluate a batch of units, in parallel if workers > 1."""
        if trains_per_unit is None:
            trains_per_unit = self.trains_per_unit
        if not units:
            return []
        if self.workers <= 1:
            return [
                self.train_config.evaluate_training_performance(u, trains_per_unit, simple_eval)
                for u in units
            ]
        eval_kwargs = self._eval_kwargs(trains_per_unit)
        payloads = [(_serialize_unit(u), eval_kwargs) for u in units]
        # chunksize keeps IPC overhead small for big populations; aim for ~4 chunks per worker.
        chunksize = max(1, len(payloads) // (self.workers * 4))
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(_evaluate_unit_worker, payloads, chunksize=chunksize))

    def _initialize_population(self):
        print("Initializing population...")
        # Build all units in the main process (cheap: just factory + random draw),
        # then evaluate them in one parallel batch.
        units = []
        for _ in range(self.pop_size):
            base_unit_class = base_unit_factory(self.num_data_obj, num_mat=self.num_op_obj)
            units.append(base_unit_class(Mat(random.randrange(base_unit_class.NUM_OPERATOR_OBJ))))
        performances = self._evaluate_units(units)
        self.population = list(zip(units, performances))
        print(f"Population initialized ({self.pop_size} units).")

    def _initialize_population_from_seed(self, units: List[Any]):
        # Evaluate caller-provided units. Used for benchmarks where the population
        # has been lifted from a smaller-dimension run rather than randomly built.
        print(f"Initializing population from {len(units)} pre-built units...")
        performances = self._evaluate_units(units)
        self.population = list(zip(units, performances))
        print("Seed population evaluated.")

    def run_evolutionary_step(self, verbose: bool = True):
        # Sort population by performance (descending)
        self.population.sort(key=lambda x: x[1], reverse=True)

        # Identify top performers
        num_top_performers = int(self.pop_size * self.top_q_percent)
        top_performers = [unit for unit, _ in self.population[:num_top_performers]]

        # Calculate performance metrics
        performances = [perf for _, perf in self.population]
        avg_performance = sum(performances) / self.pop_size
        max_unit_info = self.population[0]
        if verbose:
            print(f"Average performance: {avg_performance:.6f}, Max performance: {max_unit_info[1]:.6f}")

        validated_max_perf = self.train_config.evaluate_training_performance(
                max_unit_info[0],
                self.trains_per_unit * 5,
                simple_eval,
            )
        if verbose:
            print(f"validating max. Perf: {validated_max_perf:.4f}")


        num_new_units = self.pop_size - num_top_performers

        # Construct all new units in the main process first (mutate/splice/random
        # are cheap and operate on already-instantiated parents). Then evaluate
        # the whole batch in parallel.
        new_units: List[Any] = []
        for _ in range(num_new_units):
            if random.random() < 0.5 and len(top_performers) > 0:  # 50% mutate
                parent = random.choice(top_performers)
                new_units.append(simple_mutate(parent, self.mutation_probability))
            elif len(top_performers) >= 2:  # 50% splice
                p1, p2 = random.sample(top_performers, 2)
                new_units.append(splice_units(p1, p2))
            else:  # Not enough top performers — fall back to a completely random unit.
                BaseUnit = base_unit_factory(self.num_data_obj)
                new_units.append(BaseUnit(w=Mat(random.randrange(BaseUnit.NUM_OPERATOR_OBJ))))
        if verbose:
            print(f"Constructed {num_new_units} new units, evaluating...")

        new_performances = self._evaluate_units(new_units)
        new_population = list(zip(new_units, new_performances))

        # Re-evaluate top performers under fresh random datasets so their kept-over
        # scores aren't stale carryovers from prior generations.
        top_perf_scores = self._evaluate_units(top_performers)
        top_performers_reevaluated = list(zip(top_performers, top_perf_scores))

        # Combine top performers and new units to form the next generation
        self.population = top_performers_reevaluated + new_population

        return {
            "avg_performance": avg_performance,
            "max_performance": max_unit_info[1],
            "validated_max_performance": validated_max_perf,
        } 