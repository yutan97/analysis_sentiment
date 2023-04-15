import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
import pickle
import streamlit as st
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string
import re
import swifter
from PIL import Image

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
 
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
# Nối từ không vào từ trước nó    
def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst or 'rất' in text_lst or 'quá' in text_lst or 'gần' in text_lst or 'sai' in text_lst or 'bị' in text_lst or 'như' in text_lst or 'y' in text_lst or 'cũng' in text_lst or 'hơi' in text_lst :
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if  word in ['không','rất','quá','gần','sai','bị','như','y','cũng','hơi']:
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = ' '+ word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()
# Tách Câu
def tach_cau(text):
  new_document = ''
  for sentence in sent_tokenize(text):
    sentence = sentence.replace('.','')
    new_document = new_document + sentence + ' '
  return new_document

# Tìm từ kế nhau trong chuổi
def concat_chuoi(x,text_find):
  x = x.replace(" ","")
  text_find2 = "".join(text_find)

  if text_find2 in x:
    return "_".join(text_find)
  else:
    return ""

#Xử lý dữ liệu
def text_process(data):
  # Chuẩn bị các file
  pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
  lst_word_type = ['A','AB','V','VB','VY','R','N','M','C']
  ##LOAD EMOJICON
  file = open('Data/files/emojicon.txt', 'r', encoding="utf8")
  emoji_lst = file.read().split('\n')
  emoji_dict = {}
  for line in emoji_lst:
      key, value = line.split('\t')
      emoji_dict[key] = str(value)
  file.close()
  #################
  #LOAD TEENCODE
  file = open('Data/files/teencode.txt', 'r', encoding="utf8")
  teen_lst = file.read().split('\n')
  teen_dict = {}
  for line in teen_lst:
      key, value = line.split('\t')
      teen_dict[key] = str(value)
  file.close()
  ###############
  #LOAD TRANSLATE ENGLISH -> VNMESE
  file = open('Data/files/english-vnmese.txt', 'r', encoding="utf8")
  english_lst = file.read().split('\n')
  english_dict = {}
  for line in english_lst:
      key, value = line.split('\t')
      english_dict[key] = str(value)
  file.close()
  ################
  #LOAD wrong words
  file = open('Data/files/wrong-word.txt', 'r', encoding="utf8")
  wrong_lst = file.read().split('\n')
  dict_wrong_lst = {}
  for line in wrong_lst:
    z = line.split('\t')
    if len(z) == 2:
      key,value = line.split('\t')
    else :
      key = z[0]
      value = ''
    dict_wrong_lst[key] = str(value)
  file.close()
  #################
  #LOAD STOPWORDS
  file = open('Data/files/vietnamese-stopwords.txt', 'r', encoding="utf8")
  stopwords_lst = file.read().split('\n')
  file.close()

  # Xử lý dữ liệu
  #print('Stage.1 : Chuyển dữ liệu thành chữ thường')
  test = data.swifter.apply(lambda x: str(x).lower())
  #print(test)
  #print('Stage.2 : Xử lý dấu chấm')
  test = test.swifter.apply(lambda x: regex.sub(r'\.+', " ", x))
  #print(test)
  #print('Stage.3 : Remove and replace emoji')
  test = test.swifter.apply(lambda x: ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(x)))
  #print(test)
  #print('Stage.4 : Remove and replace teencode')
  test = test.swifter.apply(lambda x: ' '.join(teen_dict[word] if word in teen_dict else word for word in x.split()))
  #print(test)
  #print('Stage.5 : Remove and replace english')
  test = test.swifter.apply(lambda x:' '.join(english_dict[word] if word in english_dict else word for word in x.split()))
  #print(test)
  #print('Stage.6 : Remove and replace wrong word')
  test = test.swifter.apply(lambda x: ' '.join(dict_wrong_lst[word] if word in dict_wrong_lst else word for word in x.split())) #Check
  #print(test)
  #print('Stage.7 : Loai khoang trang')
  test = test.swifter.apply(lambda x: regex.sub(r'\s+', ' ', x).strip())
  #print(test)
  #print('Stage.8 : Tách câu')
  test = test.swifter.apply(lambda x: tach_cau(x))
  #print(test)
  #print('Stage.9 : Chuẩn hoá unicode')
  test = test.swifter.apply(covert_unicode) #Chuẩn hóa unicode
  #print(test)
  #print('Stage.10 : Chuẩn hoá punctuation')
  test = test.swifter.apply(lambda x: ' '.join(regex.findall(pattern,x))) #Dấu câu
  #print(test)
  #print('Stage.11 : Loại các từ lớn hơn 8')
  test = test.swifter.apply(lambda x: ' '.join(word if len(word)<8 else '' for word in x.split())) #Loại từ dài hơn 8 ký tự
  #print(test)


  # Extract feature
  #print('Stage.12.1 : Tìm cụm từ có nghĩa tiền_nào_của')
  test = test.swifter.apply(lambda x : x +" "+concat_chuoi(x,['tiền','nào','của'])) #Tìm từ ghép
  #print(test)
  #print('Stage.12.2 : Tìm cụm từ có nghĩa đáng_giá')
  test = test.swifter.apply(lambda x : x +" "+concat_chuoi(x,['đáng','giá'])) #Tìm từ ghép
  #print(test)
  #print('Stage.12.3 : Tìm cụm từ có nghĩa lấy_xu')
  test = test.swifter.apply(lambda x : x +" "+concat_chuoi(x,['lấy','xu'])) #Tìm từ ghép
  #print(test)
  #print('Stage.12.4 : Tìm cụm từ có nghĩa nhận_xu')
  test = test.swifter.apply(lambda x : x +" "+concat_chuoi(x,['nhận','xu'])) #Tìm từ ghép
  #print(test)
  #print('Stage.12.5 : Tìm cụm từ có nghĩa sản_phẩm_chất_lượng')
  test = test.swifter.apply(lambda x : x +" "+concat_chuoi(x,['sản','phẩm','chất','lượng'])) #Tìm từ ghép
  #print(test)
  #print('Stage.12.6 : Tìm cụm từ có nghĩa hàng_chất_lượng')
  test = test.swifter.apply(lambda x : x +" "+concat_chuoi(x,['hàng','chất','lượng'])) #Tìm từ ghép
  #print(test)
  #print('Stage.12.7 : Tìm cụm từ có nghĩa không_được')
  test = test.swifter.apply(lambda x : x +" "+concat_chuoi(x,['không','được'])) #Tìm từ ghép
  #print(test)
  #print('Stage.13 : Tìm từ ghep word_tokenize')
  test = test.swifter.apply(lambda x : process_special_word(word_tokenize(x, format="text")+' ai')) #Tìm từ ghép
  #print(test)
  #print('Stage.14 : Loại các từ nhỏ hơn 1')
  test = test.swifter.apply(lambda x: ' '.join(word if len(word) > 1  else '' for word in x.split())) #Loại từ dài hơn 8 ký tự
  #print(test)
  #print('Stage.15 : Phan loai tu pos_tag')
  test = test.swifter.apply(lambda x: ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(x))) #Chọn loại từ
  #print(test)
  #print('Stage.16 : Loai khoang trang')
  test = test.swifter.apply(lambda x: regex.sub(r'\s+', ' ', x).strip())
  #print(test)
  #print('Stage.17 : Loai cac stopword khong can thiet')
  test = test.swifter.apply(lambda x: ' '.join('' if word in stopwords_lst else word for word in x.split())) #remove stop word
  #print(test)
  #print('Stage.18 : Loai khoang trang')
  test = test.swifter.apply(lambda x: regex.sub(r'\s+', ' ', x).strip())
  #print(test)
  return test

# GUI
st.title("Data Science Project")
st.write("## Sentiment Analysis")

# Load models
# Đọc sentiment model
with open('model_LGR.sav', 'rb') as file:
    sentiment_model = pickle.load(file)
# Đọc count model
with open('TFIDF.sav', 'rb') as file:
    count_model = pickle.load(file)

# GUI
menu = ["Giới thiệu", "Dự đoán"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Giới thiệu':
    st.image("sentiment_analysis.jpg")
    st.markdown('<div style="text-align: justify;">Thu thập thông tin phản hồi của khách hàng là một cách tuyệt vời giúp cho các doanh nghiệp hiểu được điểm mạnh, điểm yếu trong sản phẩm, dịch vụ của mình; đồng thời nhanh chóng nắm bắt được tâm ký và nhu cầu khách hàng để mang đến cho họ sản phẩm, dịch vụ hoàn hảo nhất.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">Ngày nay, với sự phát triển vượt bậc của khoa học và công nghệ, đặc biệt là sự bùng nổ của Internet với các phương tiện truyền thông xã hội, thương mại điện tử,... đã cho phép mọi người không chỉ chia sẻ thông tin trên đó mà còn thể hiện thái độ, quan điểm của mình đối với các sản phẩm, dịch vụ và các vấn đề xã hội khác. Vì vậy mà Internet đã trở lên vô cùng quan trọng và là nguồn cung cấp một lượng thông tin vô cùng lớn và quan trọng.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">Thông qua những dữ liệu được thu thập từ Shopee, các nhà cung cấp dịch vụ cũng có thể sử dụng những nguồn thông tin này để đánh giá về sản phẩm của mình, từ đó có thể đưa ra những cải tiến phù hợp hơn với người dùng, mang lại lợi nhuận cao hơn, tránh các rủi ro đáng tiếc xảy ra. Đặc biệt, khi 1 doanh nghiệp có 1 sản phẩm mới ra mắt thị trường thì việc lấy ý kiến phản hồi là vô cùng cần thiết.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">Sau các phân tích và xây dựng các model theo từng ngành hàng có trong bộ dữ liệu bao gồm Điện Thoại & Phụ Kiện, Máy Ảnh & Máy Quay Phim, Máy Tính & Laptop, Mẹ & Bé, Nhà Cửa & Đời Sống, Sắc Đẹp, Thiết Bị Điện Tử, Thời Trang Nam, Thời Trang Nữ nhằm mang lại kết quả tối ưu cho việc đánh giá phản hồi khách hàng với Machine learning. Kết quả các ngành hàng đều cho kết quả khả quan với accuracy ~ 0.80 với model MultinomialNB và cùng một bộ parameter alpha = 10.0, class_prior = None, fit_prior = False.</div>', unsafe_allow_html=True)
    # st.write("""
    # Thu thập thông tin phản hồi của khách hàng là một cách tuyệt vời giúp cho các doanh nghiệp hiểu được điểm mạnh, điểm yếu trong sản phẩm, dịch vụ của mình; đồng thời nhanh chóng nắm bắt được tâm ký và nhu cầu khách hàng để mang đến cho họ sản phẩm, dịch vụ hoàn hảo nhất.
    # """)
    # st.write("""
    # Ngày nay, với sự phát triển vượt bậc của khoa học và công nghệ, đặc biệt là sự bùng nổ của Internet với các phương tiện truyền thông xã hội, thương mại điện tử,... đã cho phép mọi người không chỉ chia sẻ thông tin trên đó mà còn thể hiện thái độ, quan điểm của mình đối với các sản phẩm, dịch vụ và các vấn đề xã hội khác. Vì vậy mà Internet đã trở lên vô cùng quan trọng và là nguồn cung cấp một lượng thông tin vô cùng lớn và quan trọng.
    # """)
    # st.write("""
    # Thông qua những dữ liệu được thu thập từ Shopee, các nhà cung cấp dịch vụ cũng có thể sử dụng những nguồn thông tin này để đánh giá về sản phẩm của mình, từ đó có thể đưa ra những cải tiến phù hợp hơn với người dùng, mang lại lợi nhuận cao hơn, tránh các rủi ro đáng tiếc xảy ra. Đặc biệt, khi 1 doanh nghiệp có 1 sản phẩm mới ra mắt thị trường thì việc lấy ý kiến phản hồi là vô cùng cần thiết.
    # """)
    # st.write("""
    # Sau các phân tích và xây dựng các model theo từng ngành hàng có trong bộ dữ liệu bao gồm Điện Thoại & Phụ Kiện, Máy Ảnh & Máy Quay Phim, Máy Tính & Laptop, Mẹ & Bé, Nhà Cửa & Đời Sống, Sắc Đẹp, Thiết Bị Điện Tử, Thời Trang Nam, Thời Trang Nữ nhằm mang lại kết quả tối ưu cho việc đánh giá phản hồi khách hàng với Machine learning. Kết quả các ngành hàng đều cho kết quả khả quan với accuracy ~ 0.80 với model MultinomialNB và cùng một bộ parameter alpha 10.0, class_prior None, fit_prior False.
    # """)

elif choice == 'Dự đoán':
    flag = False
    lines = None
    type = st.selectbox("Chọn phương thức muốn phân tích:", options=("Tải lên tệp *.csv", "Nhập nội dung mới"))
    if type=="Tải lên tệp *.csv":
        # Upload file
        uploaded_file = st.file_uploader("Chọn tệp", type=['csv'])
        if uploaded_file is not None:
            lines = pd.read_csv(uploaded_file)
            lines = text_process(lines['comment'])
            x_new = count_model.transform(lines)        
            y_pred_new = sentiment_model.predict(x_new)
            lines['prediction'] = y_pred_new
            st.write("Kết quả phân tích:")
            st.dataframe(lines['prediction'])
    if type=="Nhập nội dung mới":
        with st.form(key='my_form'):
	        review = st.text_input(label='Nhập nội dung cần phân tích:')
	        submit_button = st.form_submit_button(label='Phân tích')
            #form = st.form(key='my_form')
            #review = st.text_area(label="Nhập nội dung cần phân tích:")
            #submit_button = st.form_submit_button(label='Phân tích')
        if review!="":
            lines = pd.DataFrame({'comment':[review]})
            lines = text_process(lines['comment'])
            x_new = count_model.transform(lines)
            y_pred_new = sentiment_model.predict(x_new)
            if y_pred_new == "negative":
                st.write("""
                Khách hàng phản hồi không tốt về sản phẩm này, cần cải thiện.
                """)
                neg = Image.open("negative.jpg")
                neg = neg.resize((400,400))
                st.image(neg, width = 250)
            elif y_pred_new == "neutral":
                st.write("""
                Khách hàng có cảm nhận bình thường về sản phẩm này.
                """)
                neu = Image.open("neutral.png")
                neu = neu.resize((400,400))
                st.image(neu, width = 250)
            elif y_pred_new == "positive":
                st.write("""
                Chúc mừng bạn, bạn có một phản hồi tốt về sản phẩm này.
                """)
                pos = Image.open("positive.jpg")
                pos = pos.resize((400,400))
                st.image(pos, width = 250)
