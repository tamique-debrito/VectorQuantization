This is an initial repository for an idea I came up with for a class in my senior year of college (6.UAT in Spring 2022).
The summary is that it's a form of quantization for neural networks that, if effective (and hasn't already been done), could improve train and inference efficiency.
See the "Proposal Talk.pdf" (a couple relevant slides beloy). This presentation was developed along with my classmate Noah Faro. My main contribution was the technical idea definition and plan, while Noah contributed background research, slide design, presentation structure, and probably other things I'm forgetting.

The code here is a quick proof of concept that shows that quantizing this way can use SGD to learn a single linear layer to fit a dataset.

![image](https://github.com/user-attachments/assets/cff8dd23-0333-4865-b734-4bb5d64d5048)

![image](https://github.com/user-attachments/assets/311d4148-a0b1-4c80-a368-ea08e0651c59)

