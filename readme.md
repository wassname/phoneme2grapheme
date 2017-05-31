# phon2graph

Using a sequence 2 sequence model with attention to convert from pronunciation to spelling.


## results

It reaches a character error rate of <20%, and here are the results (lighter letters show where the model was uncertain). The machine has made some reasonable mistakes.


<table><tbody></tbody><thead><tr><th>pronunciation</th><th>guess</th><th>spelling</th></tr></thead><tbody><tr><td><span class="1.0" style="color:rgba(0,0,0,1.0)">k</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">ɑ</span><span style="color:rgba(0,0,0,1.0)">k</span><span style="color:rgba(0,0,0,1.0)">ə</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)">ˌ</span><span style="color:rgba(0,0,0,1.0)">ɔ</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">d</span></td><td><span style="color:rgba(0,0,0,0.9)">c</span><span style="color:rgba(0,0,0,0.6)">o</span><span style="color:rgba(0,0,0,0.7)">c</span><span style="color:rgba(0,0,0,0.5)">k</span><span style="color:rgba(0,0,0,0.4)">s</span><span style="color:rgba(0,0,0,0.6)">o</span><span style="color:rgba(0,0,0,0.5)">i</span><span style="color:rgba(0,0,0,0.5)">d</span><span style="color:rgba(0,0,0,0.8)"> </span><span style="color:rgba(0,0,0,0.9)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">c</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">u</span><span style="color:rgba(0,0,0,1.0)">c</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)">o</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">d</span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">ɹ</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">æ</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)">f</span><span style="color:rgba(0,0,0,1.0)">ˌ</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">d</span></td><td><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">r</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,0.9)">s</span><span style="color:rgba(0,0,0,0.6)">s</span><span style="color:rgba(0,0,0,0.5)">f</span><span style="color:rgba(0,0,0,0.5)">i</span><span style="color:rgba(0,0,0,0.6)">l</span><span style="color:rgba(0,0,0,0.7)">d</span><span style="color:rgba(0,0,0,0.6)">d</span></td><td><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">r</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)">f</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">d</span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">f</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">æ</span><span style="color:rgba(0,0,0,1.0)">ʃ</span><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">̩</span><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">f</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,0.8)">s</span><span style="color:rgba(0,0,0,0.7)">h</span><span style="color:rgba(0,0,0,0.7)">b</span><span style="color:rgba(0,0,0,0.5)">l</span><span style="color:rgba(0,0,0,0.5)">b</span><span style="color:rgba(0,0,0,0.5)"> </span><span style="color:rgba(0,0,0,0.8)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">f</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)">h</span><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">u</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">h</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">ɛ</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">ə</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">h</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,0.9)">s</span><span style="color:rgba(0,0,0,0.9)">t</span><span style="color:rgba(0,0,0,0.6)">i</span><span style="color:rgba(0,0,0,0.5)">a</span><span style="color:rgba(0,0,0,0.8)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">h</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">k</span><span style="color:rgba(0,0,0,1.0)">ɹ</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">æ</span><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,0.7)">c</span><span style="color:rgba(0,0,0,0.9)">r</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,0.9)">b</span><span style="color:rgba(0,0,0,0.5)">y</span><span style="color:rgba(0,0,0,0.6)">y</span><span style="color:rgba(0,0,0,0.9)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">c</span><span style="color:rgba(0,0,0,1.0)">r</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">y</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">ɹ</span><span style="color:rgba(0,0,0,1.0)">ə</span><span style="color:rgba(0,0,0,1.0)">d</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">u</span><span style="color:rgba(0,0,0,1.0)">s</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,0.8)">t</span><span style="color:rgba(0,0,0,1.0)">r</span><span style="color:rgba(0,0,0,0.6)">o</span><span style="color:rgba(0,0,0,0.9)">d</span><span style="color:rgba(0,0,0,0.7)">u</span><span style="color:rgba(0,0,0,0.4)">s</span><span style="color:rgba(0,0,0,0.5)">e</span><span style="color:rgba(0,0,0,0.7)"> </span><span style="color:rgba(0,0,0,0.9)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">r</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">d</span><span style="color:rgba(0,0,0,1.0)">u</span><span style="color:rgba(0,0,0,1.0)">c</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">ə</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">u</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">ə</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,0.9)">t</span><span style="color:rgba(0,0,0,0.7)">a</span><span style="color:rgba(0,0,0,0.8)">l</span><span style="color:rgba(0,0,0,0.6)">u</span><span style="color:rgba(0,0,0,0.8)">l</span><span style="color:rgba(0,0,0,0.6)">a</span><span style="color:rgba(0,0,0,0.5)">a</span><span style="color:rgba(0,0,0,0.9)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">u</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">h</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">n</span><span style="color:rgba(0,0,0,1.0)">̩</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,0.7)">e</span><span style="color:rgba(0,0,0,0.4)">a</span><span style="color:rgba(0,0,0,0.6)">t</span><span style="color:rgba(0,0,0,0.4)">n</span><span style="color:rgba(0,0,0,0.6)"> </span><span style="color:rgba(0,0,0,0.9)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">o</span><span style="color:rgba(0,0,0,1.0)">n</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">h</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">ɑ</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">ɪ</span><span style="color:rgba(0,0,0,1.0)">t</span><span style="color:rgba(0,0,0,1.0)">ʃ</span><span style="color:rgba(0,0,0,1.0)">ɛ</span><span style="color:rgba(0,0,0,1.0)">k</span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,0.9)">h</span><span style="color:rgba(0,0,0,0.6)">a</span><span style="color:rgba(0,0,0,0.9)">l</span><span style="color:rgba(0,0,0,0.7)">i</span><span style="color:rgba(0,0,0,0.6)">c</span><span style="color:rgba(0,0,0,0.5)">c</span><span style="color:rgba(0,0,0,0.5)">k</span><span style="color:rgba(0,0,0,0.7)"> </span><span style="color:rgba(0,0,0,0.9)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">h</span><span style="color:rgba(0,0,0,1.0)">o</span><span style="color:rgba(0,0,0,1.0)">l</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,1.0)">c</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,1.0)">k</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr><tr><td><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">d</span><span style="color:rgba(0,0,0,1.0)">ˈ</span><span style="color:rgba(0,0,0,1.0)">ɔ</span><span style="color:rgba(0,0,0,1.0)">i</span><span style="color:rgba(0,0,0,1.0)">ə</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,0.6)">b</span><span style="color:rgba(0,0,0,0.6)">e</span><span style="color:rgba(0,0,0,0.5)">d</span><span style="color:rgba(0,0,0,0.4)">o</span><span style="color:rgba(0,0,0,0.4)">o</span><span style="color:rgba(0,0,0,0.4)">y</span><span style="color:rgba(0,0,0,0.8)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td><td><span style="color:rgba(0,0,0,1.0)">b</span><span style="color:rgba(0,0,0,1.0)">e</span><span style="color:rgba(0,0,0,1.0)">d</span><span style="color:rgba(0,0,0,1.0)">o</span><span style="color:rgba(0,0,0,1.0)">y</span><span style="color:rgba(0,0,0,1.0)">a</span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span><span style="color:rgba(0,0,0,1.0)"> </span></td></tr></tbody></table>

## installation

Install the requirements `pip install --upgrade -r requirements.txt`

Then start a jupyter notebook and open main.ipynb
