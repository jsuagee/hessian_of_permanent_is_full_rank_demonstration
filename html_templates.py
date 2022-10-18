

original_html = r'''<html>
<head>
    <title>Reduction Sequence</title>
    <style>
        .message {
                  text-align: center;
                  display: none;
        }
        .dataframe { 
                font-size: 11pt; 
                font-family: Arial; 
                border-collapse: collapse; 
                border: 1px solid silver;
                text-align: right;
                margin-left: auto;
                margin-right: auto;
        }
        .controls-container {
          text-align: center;
          margin-right: auto;
        }
    </style>
</head>
<body>
    <div class="message-container">
    </div>
    <div class="slideshow-container">
    </div>
    <div class="controls-container">
    </div>
</body>
</html>'''

script = '''
let slideIndex = 1;
showSlides(slideIndex);

function plusSlides(n) {
  showSlides(slideIndex += n);
}

function showSlides(n) {
  let i;
  let slides = document.getElementsByClassName("slide");
  let messages = document.getElementsByClassName("message");

  if (n > slides.length) {slideIndex = 1}
  if (n < 1) {slideIndex = slides.length}
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
    messages[i].style.display = "none";
  }
  slides[n-1].style.display = "block";
  messages[n-1].style.display = "block";
}
'''






