### GenAI Lab 3 NLP Report
#### by Roman Kovalchuk

### Environment:

For environment for this task I've used Google Colab with A100 gpu with 40 GB of GRAM. It provides enough memory to do this task. Priorly this is due to the minimal number of output token for the ZNO dataset, as we're only required to select a single answer from 5 choices in our response.

### Libraries:

The choice of libraries is pretty ordinal, I've chosen transformers, trl, peft and huggingface hub to accomplish this task.

### Model Choice:

Deriving from the past experience for Ukrainian NLP tasks with LLMs during this course, I've settled on `unsloth/Qwen2.5-32B-bnb-4bit` model with 32B of parameters, quantizated to 4 bits precision with bitsandbites library, saved from huggingface.
The neat point is that we can very simply reuse this model, as we do not need to do any quantization rpocedures necessary to fit this model on our GPU both for inference, and specifically training.

Luckily, A100 GPU does support flash_attention_2, bnb quantization and other neat techniques, that allow us to fit this high parameter model in our GPU for training.

With other model I've tested for previous assignments and other out of course tasks, without fine-grained fine-tuning data, it might be hard for the OpenSource LLM to professionally navigate and eloquently use Ukrainian.

Also with this model, many of my teammates achieved high results, and I've not used it for previous assignments, so it made me curious to use it for this one.

### Instruction Tuning Templates:

For the training I use following template, derived from huggingface responses from the dev team:

```python
"<|im_start|>system\Ти є розумним асистентом що дає відповіді на тести українською. Відповіді повинні бути зазначені у форматі одного варіанту серед наданих: А, Б, В, Г або ж Д. Твоя відповідь має включати лише варіант відповіді на запитання з тесту.<|im_end|>\n" \
"<|im_start|>user\n{question}\n{answers}<|im_end|>\n" \
"<|im_start|>assistant\n{correct_answers}<|im_end|>\n"
```

P.S. Note, different OpenSource LLMs, use different templates, and in order to achieve good results and convergence, we got to rely on them.

Furthermore, however, I could have experiment with different IT templates in this task, due to limited time I've had, and most importantly, little to no benefit other researchers achieve during the IT template selection, I've decided to not experiment further in this direction. My thinking was, I need to put all the most crucial information down there, so the model does converge faster. Plus, the variables I need for any training with any IT templates: "task description (first line)", "input question and possible answers (second line - feature)", and "correct answers (third line, which is our target data)"

For the inference stage, the template is similar, with the only difference being is the absence of target, as the LLM got to generate us one.

```python
"<|im_start|>system\Ти є розумним асистентом що дає відповіді на тести українською. Відповіді повинні бути зазначені у форматі одного варіанту серед наданих: А, Б, В, Г або ж Д. Твоя відповідь має включати лише варіант відповіді на запитання з тесту.<|im_end|>\n"\
"<|im_start|>user\n{question}\n{answers}<|im_end|>"\
"<|im_start|>assistant"
```

### Number of generated tokens in SFT parameters and Tokenizer set-up

Whilst, this is an important task in LLM fine-tuning, as we've got to play around with the size of sample, using ConstantLenghtDataset, batching, GPU memory, and most importantly LLM capabilities in processing and generating output data, this task is fairly simple, so I've set input to 250 tokens, and output - `max_new_tokens` to be 20, just to be sure.

### Lora adapter size

Nowadays, a super popular option, other than lora alpha being twice size of lora rank, is lora alpha being sized at 32, and r=16; So I've decied to put a larger lora adapter with size of lora_alpha being=64, and lora_r_rank=32.\

However, In my experience, those changes make a little difference depending on the complexity of downstream task and structure of generated (target) data, influences most and foremost the speed of processing, and the convergence time, achieving the same results with less time.

Other parameters I've took from default fine-tuning notebooks from guide for Qwen2.5 32B fine tuning notebooks.

### Artifacts

The datasets with no alterations from originals can be accessed from GDrive:
- Train: https://drive.google.com/file/d/1S7Sd0ldQHWxJX87l0RDQfj-YZwlK0R7H/view?usp=sharing
- Test: https://drive.google.com/file/d/1Y7EpfUYEvCTAfptjlaqPPLi5LUoyEHVS/view?usp=sharing

The inference data can be found on GDrive by link:
- `submission.csv` Inference (Preds): https://drive.google.com/file/d/1w7HFlC0Nt3fN6d9vTnD3w27Zx6gIDH0L/view?usp=sharing
- `submission_postprocessed.csv` Post-inference (extracted answers with regex, final): https://drive.google.com/file/d/1Kpb0kYHn9gT1gvfi-2aUT0ABxqStdEvd/view?usp=sharing 

Models were uploaded with adapters to:
- Huggingface account: https://huggingface.co/reennon

### Results:

We've achieved very splendid results in my opinion:

![Screenshot 2025-02-02 at 21.28.26.png](images%2FScreenshot%202025-02-02%20at%2021.28.26.png)

# Postprocessing:

I've used `parsing_script` to extract raw answers from the model, as it includes some uaxiliary data LLMs tend to generate, over the answer, with simple regex, the option after the word `assistant` or the first option, if LLM predicted many, gets treated as final answer. By so response from our Qwen model:

```text
system\Ти є розумним асистентом що дає відповіді на тести українською. Відповіді повинні бути зазначені у форматі одного варіанту серед наданих: А, Б, В, Г або ж Д. Твоя відповідь має включати лише варіант відповіді на запитання з тесту.
user
Прочитайте речення *(цифра позначає попередній розділовий знак).*


*Як стверджують філософи й соціологи****,(1)*** *цивілізованість світу вимірюють* *книжками****,(2)*** *саме вони є беззаперечними символами мудрості й честі****:(3)*** *президенти країн****,(4)*** *присягаючи нації****,(5)*** *кладуть руку саме на книгу****,(6)*** *а не* *на пощерблену в битвах древню шаблю.*


НЕПРАВИЛЬНО обґрунтовано вживання розділових знаків у рядку
[{'marker': 'А', 'text': 'двокрапка 3 – між частинами складного речення, що поєднані безсполучниковим зв’язком'}, {'marker': 'Б', 'text': 'кома 2 – між частинами складного речення, що поєднані безсполучниковим зв’язком'}, {'marker': 'В', 'text': 'коми 4, 5 – при відокремленій обставині'}, {'marker': 'Г', 'text': 'кома 6 – при однорідних членах речення'}, {'marker': 'Д', 'text': 'кома 1 – між частинами складного речення, що поєднані підрядним зв’язком'}]
assistant
['А' 'В' 'Г' 'Д' 'Б' 'А' 'В
```
becomes
```text
A
```

### Conclusions, and possible improvements/experiments

The Qwen2.5 32B despite being quantized to 4bit precision using bitsandbites by author, achieves a comparable results, despite being trained only for a few epochs on our prime (only ZNO dataset, without any auxiliary ones)\

I would want to experiment with post-tuning techniques: like dynamic few-shots, prompt engineering, etc with the trained model, as it's interesting whether it still remain capable of being prompted after our fine-tuning (catastrophic forgetting)

We can also play with tuning parameters, dataset parameters, preprocessing, but I doubt that would yield in significantly better results either without an addition of new relevant data (like wikipedia) or a good few-shot technique, after the fine-tuning.

P.S. if you intend to run this notebook, please change sensitive variables like huggingface keys from REDACTED to yours. 