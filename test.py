from rouge import Rouge
text = 'Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump'

text2 = ['Tổng thống Mỹ Donald Trump sẽ đến Việt Nam vào tháng 11 tới đây', 'Donald Trump bú cặc chó']

rouge = Rouge()

score = rouge.get_scores(text, text2)

