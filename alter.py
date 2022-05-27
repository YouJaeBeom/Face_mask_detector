import requests

#curl -X POST -H "Content-Type: application/json" \
#-d "{\"text\":\"빨리 끝내고 자자. :rocket:\"}" \
#https://hooks.slack.com/services/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
def alert(message):
    r = requests.post('https://hooks.slack.com/services/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', json={"text": "{}:red_circle:".format(message)}, headers={"Content-Type": "application/json"})
    print(r.content)
    return True
