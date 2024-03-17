# from GPTme.pipes.local_crag_websearch import run_crag_app
from GPTme.pipes.local_crag import run_crag_app

def CRAG_QA():

    print("Hello I am here to help you find answers from your documentation.\nPlease let me know what you would like to know. to end our conversation please enter 'bye'")
    while True:
        question = input("User entry - ")
        if question != "bye":
            result = run_crag_app(question)
            print(result)
        else:
            print('bye bye')
            return
    
CRAG_QA()