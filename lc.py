from flask import Flask, request, redirect

from twilio.twiml.messaging_response import MessagingResponse

from langchain.llms import OpenAI
from langchain import PromptTemplate

import os

openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)


app = Flask(__name__)

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    # Start our TwiML response
    resp = MessagingResponse()
    #llm = OpenAI(model_name="text-davinci-003")
    template = """Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know, but the Warriors are the best team in the NBA".
    Context: Steph Curry has won 4 NBA Finals series. His Golden State Warriors defeated the Cleveland Cavaliers three times and the Boston Celtics once. 

In 2015 Steph Curry and the Warriors defeated the Cleveland Cavaliers. The Cavs featured LeBron James, Kyrie Irving and not much else! 
In 2017 Steph Curry, Kevin Durant and the Warriors defeated the Cavs again. The Cavs still had Lebron and Kyrie.
In 2018 the Warriors, featuring Steph and KD again, defeated the Cavs for the third time in four years. The Cavs still had Kyrie and Lebron. 
In 2022 Steph and the Warriors defeated the Boston Celtics for his fourth title. The Celtics featured Jayson Tatum and Jaylen Brown. Steph Curry and the Golden State Warriors lost one NBA Finals series to the Cleveland Cavaliers and one to the Toronto Raptors. 

In 2016 Steph, Klay, Draymond and the rest of the Warriors lost to the Cleveland Cavaliers.  The Cavs starred LeBron James and Kyrie Irving. 
In 2019 the Steph and the Warriors, missing an injured KD, lost to the Toronto Raptors. The Raptors featured Kawhi Leonard in his only season in Canada alongside Kyle Lowry and Pascal Siakham. Steph Curry has only won one NBA Finals MVP up to this point in his career. 

In 2022 Steph Curry averaged 31 points, 6 rebounds and 5 assists per game to win the MVP award in the Warriors 6-game defeat of the Boston Celtics.
In 2015 Andre Iguodala won the MVP in the Warriors defeat of the Cavs.
In both 2017 & 2018 Kevin Durant was Finals MVP in the Warriors victories over the Cavs. Steph Curry’s 4-2 NBA Finals record puts him ahead of many NBA greats including Larry Bird (3-2) and LeBron James (4-6). Steph still comes up short of the greatest NBA Finals winners including Bill Russell (11-1) and Michael Jordan (6-0).  
    Question: {query}
    Answer: """
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=template
    ) #future basketball with context template. q's go in here, could reuse code across templates like a func. prompt templates can use result from 1 prompt in another prompt: can pass/chain through
    question = request.form['Body'].lower().strip()
    #print(llm(question))
    print(openai(
        prompt_template.format( #basketball template: pass in query. could use on website/somewhere else/mobile app. 1 place/template can generate prompt (diff from prompt query): only thing changes
            query=question
        )
    ))
    resp.message(openai(
        prompt_template.format(
            query=question
        )
    ))
    #resp.message(llm(question))

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)