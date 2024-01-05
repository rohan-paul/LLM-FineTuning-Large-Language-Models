## `HfArgumentParser` - How it works

The `HfArgumentParser` class is provided in Hugging Face’s transformers library. This class takes one or more data classes and turns their data attributes into command-line arguments of an ArgumentParser object.

📌 In above code, the `TrainingArguments` data class is defined with various fields like `local_rank`, `per_device_train_batch_size`, `model_name`, etc. These fields are annotated with types and optional metadata.

📌 When the `HfArgumentParser` is instantiated with `TrainingArguments` class, it introspects this class and automatically generates corresponding command-line arguments. Therefore, when you run your script from the command line, you can specify values for these arguments, like `--local_rank 0`, `--per_device_train_batch_size 8`, and so on.

Each field in the data classes becomes a command-line argument. The type annotations and default values in the data classes are used to determine the type and default value of each command-line argument.

Upon execution, the parser parses the command-line arguments and instantiates objects of the provided data classes, filling each field with the corresponding value from the command line.

--------

📌 `torch.cuda.get_device_capability()` returns the major and minor cuda capability of the current device

📌 Where `major` and `minor` are integers representing the major and minor revision numbers of the compute capability.
The major revision number (the first element of the tuple) is the most relevant for determining general capabilities. Higher numbers typically indicate support for more advanced features and better performance.

📌 The compute capability is a version number that corresponds to a specific GPU architecture's features. Here, the major version number of the compute capability is extracted and checked.

📌 For example, if major is 7 and minor is 5, cuda capability is 7.5.

📌 And here in this code, if the major version number major is greater than or equal to 8, it indicates that the GPU supports bfloat16. This is because NVIDIA GPUs with a compute capability of 8.0 or higher (Ampere architecture and newer) include support for bfloat16 operations, which are designed to provide near-float32 level of precision for deep learning