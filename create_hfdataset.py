import datasets
import pandas as pd

def get_prompt(Input_text, out_text, with_ans, examples=[], answers=[]):
    
    prompt3 = "Extract the following information from Real Context and answer the following questions without explanation: \n" + \
        f"Real Context: ### {Input_text} ### \n" + \
        "### Question 1: Is a water heater mentioned in the context? (A) No (B) Yes \n" + \
        "### Question 2: Number of water heaters mentioned? \n" + \
        "### Question 3: If applicable, what type of water heater is there? (A) Not applicable (B) Electric (C) Gas (D) Tankless (E) Other \n" + \
        "### Question 4: If applicable, is the water heater new? (A) No or Not applicable (B) Yes \n" + \
        "Here are example contexts and answers for the above questions. Don't mention the examples in your answer. \n"
    #    "### Start Examples ### \n"
    #for i in range(len(examples)):
    #    prompt3 += f"Example Context: ### {examples[i]} ### \n"
    #    prompt3 += f"### Example Answers: {answers[i]} \n"
    #prompt3 += "### End Examples ### \n"
    prompt3 += " ### Default Answer Format: Question 1: (A) Question 2: (0) Question 3: (A) Question 4: (A) \n ### Answers: "
    labels = [x for x in out_text if x is not None]
    labels = ['0' if "N/A" in x else x for x in labels ]
    out_gen = f"Question 1: ({chr(int(labels[0]) + 65)}) Question 2: ({labels[1]}) Question 3: ({chr(int(labels[2]) + 65)}) Question 4: ({chr(int(labels[3]) + 65)})"
    if with_ans == True:
        prompt3 += out_gen
    #print(prompt3)
    return prompt3

austin_df = pd.read_pickle('GPT_zeroshot_categories.df')
examples = []
answers = []
austin_df['zeroprompt'] = austin_df.apply(lambda x: get_prompt(x["description"], x["Water Heater"], True, examples, answers), axis=1)
austin_df['zeroprompt_noans'] = austin_df.apply(lambda x: get_prompt(x["description"], x["Water Heater"], False, examples, answers), axis=1)
austin_dataset = datasets.Dataset.from_pandas(pd.DataFrame(austin_df[["zeroprompt", 'zeroprompt_noans']], columns=['zeroprompt', 'zeroprompt_noans']))
austin_dataset.save_to_disk("austin_test_V3.hf")