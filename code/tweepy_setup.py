import tweepy

api_key = "k4aAgrvgf51StMMCMav6a4BrQ"
api_secret = "JMIuEjbSUcFCg89NSiRdEhN2F5nBUAKFGsUdo9r91zGqQhEMYX"
bearer_token = r"AAAAAAAAAAAAAAAAAAAAAFpevQEAAAAAz4AV0u%2FnJxoJ%2BXje7f%2ByrN8pSLY%3Db5Y35gvUO9U2MBDzt8Hwbony29OdkKZE6xAYtHNoig1WsoVCSy"
access_token = "1821233002464280576-QBbt1LW94GooGqcrrPdKLepOQRnSGS"
access_token_secret = "RwephwK46qgJMerRMLkFDiRiJCw44eSTkqcHZWJFgcGgA"

client = tweepy.Client(bearer_token, api_key, api_secret, access_token, access_token_secret)

# Creating API instance. This is so we still have access to Twitter API V1 features
auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Creating a tweet to test the bot

username = 'Sumitha2453'

# Post a tweet mentioning the user
tweet_text = f"@{username} Hello!If you're feeling like you need support, please reach out to a us. You are not alone"
client.create_tweet(text=tweet_text)


