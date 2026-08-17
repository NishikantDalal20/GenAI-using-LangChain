from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation',
    pipeline_kwargs=dict(temperature= 0.5, max_new_tokens=100)
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)


# dont run it

# run command:python ./2.ChatModels/4_chatmodel_hf_local.py