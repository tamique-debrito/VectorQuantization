This is an initial repository for an idea I came up with for a class in my senior year of college (6.UAT).
The summary is that it's a form of quantization for neural networks that, if effective (and hasn't already been done), could improve train and inference efficiency.
See the "Proposal Talk.pdf" (this presentation was developed along with my classmate Noah Faro. My main contribution was the technical idea definition, while Noah contributed background research, slide design, and presentation structure)

The code here is a quick proof of concept that shows that quantizing this way can use SGD to learn a single linear layer to fit a dataset.