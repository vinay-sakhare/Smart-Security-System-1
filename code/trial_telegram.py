import telegram


#token that can be generated talking with @BotFather on telegram
my_token = '1165920247:AAEwWb1zxHoae9gjqJhgxkcufPfZQgXvsWI'
"""
Send a message to a telegram user or group specified on chatId
chat_id must be a number!
"""
bot = telegram.Bot(token=token)
bot.sendMessage(chat_id=387654397, text=msg)
