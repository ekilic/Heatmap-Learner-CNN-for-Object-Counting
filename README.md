# Heatmap Learner Convolutional Neural Network for Object Counting and Localization
Member: <a href="https://github.com/ekilic/"> Ersin KILIÇ  </a> <a href = "mailto:ersinkilic@erciyes.edu.tr">(E-mail)</a></br>


<blockquote>
<p><a href="https://link.springer.com/article/10.1007/s12652-021-03377-5" rel="nofollow"><strong>An accurate car counting in aerial images based on convolutional neural networks
</strong></a>,<br>
Ersin Kilic, Serkan Öztürk<br>
<em>SOTA of CARPK(<a href="https://paperswithcode.com/sota/object-counting-on-carpk" rel="nofollow">SOTA of CARPK</a>)</em></p>
</blockquote>

<h2>Description</h2>

This project aims to implement a simple and effective single-shot detector model to detect and count the cars in aerial images. 


<h2>Results</h2>

Experiments on the <a href="https://lafi.github.io/LPN/">CARPK</a> car dataset have shown the state-of-the-art counting and localizing performance of the proposed method compared with existing methods.

<table>
<tbody>
<tr>
<td><b>CARPK Dataset</b></td>
<td><b>MAE</b></td>
<td><b>RMSE</b></td>
</tr>
 <tr>
<td>LPN&nbsp;[1]</td>
<td>23.80</td>
<td>36.79</td>
</tr>
 <tr>
<td>RetinaNet[2]</td>
<td>16.62</td>
<td>22.30</td>
</tr>
 <tr>
<td>YOLOv3[3]</td>
<td>7.92</td>
<td>11.08</td>
</tr>
 <tr>
<td>IoUNet[4]</td>
<td>6.77</td>
<td>8.52</td>
</tr>
 <tr>
<td>VGG-GAP-HR[5]</td>
<td>7.88</td>
<td>9.30</td>
</tr>
<tr>
<tr>
<td>GSP224[6]</td>
<td>5.46</td>
<td>8.09</td>
</tr>
 <tr>
<td>SA+CF+CRT[7]</td>
<td>5.42</td>
<td>7.38</td>
</tr>
 <tr>
<td>GANet (VGG-16)[8]</td>
<td>5.80</td>
<td>6.90</td>
</tr>
<td>Ours (VGG-16)</td>
<td><b>2.12</b></td>
  <td><b>3.02</b></td>
</tr>
</tbody>
</table>


<h2>Sample Localization Results</h2>

<table>
<tbody>
<tr>
<td>Original</td>
<td>Heat map</td>
<td>Peak map</td>
</tr>
<tr>
<td><img src="results/457.png"/></td>
<td><img src="results/heatmap-457.png"/></td>
<td><img src="results/peakmap-457.png"/></td>
</tr>
<tr>
<td><img src="results/456.png"/></td>
<td><img src="results/heatmap-456.png"/></td>
<td><img src="results/peakmap-456.png"/></td>
</tr>
</tbody>
</table>

<h2>Test</h2>
1- Download pretrained model (<a href="https://drive.google.com/file/d/1a1MX70msKq_gLgquD34TuUWyMeXHEJ2e/view?usp=sharing"> CARPK (downsampling ratio = 8) </a>)
2- Start visdom and browse (http://localhost:8097/)
3- Run test.py

<h2>References</h2>
[1] M. Hsieh, Y. Lin, W. H. Hsu, Drone-based object counting by spatially regularized regional proposal network, CoRR abs/1707.05972. arXiv: 1707.05972. </br>
[2] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollar, Focal loss for dense object detection, IEEE Transactions on Pattern Analysis and Machine Intelligence (2018) 1–1doi:10.1109/tpami.2018.2858826.</br>
[3] J. Redmon, A. Farhadi, Yolov3: An incremental improvement, CoRR abs/1804.02767. arXiv:1804.02767.</br>
[4] E. Goldman, R. Herzig, A. Eisenschtat, O. Ratzon, I. Levi, J. Goldberger, T. Hassner, Precise detection in densely packed scenes, CoRR abs/1904.00853. arXiv:1904.00853.</br>
[5] S. Aich, I. Stavness, Improving object counting with heatmap regulation, CoRR abs/1803.05494. arXiv:1803.05494.</br>
[6] S. Aich, I. Stavness, Object counting with small datasets of large images, CoRR abs/1805.11123. arXiv:1805.11123.</br>
[7] W. Li, H. Li, Q. Wu, X. Chen, K. N. Ngan, Simultaneously detecting and counting dense vehicles from drone images, IEEE Transactions on Industrial Electronics 66 (12) (2019) 9651–9662. doi:10.1109/tie.2019.2899548.</br>
[8] Y. Cai, D. Du, L. Zhang, L. Wen, W. Wang, Y. Wu, S. Lyu, Guided attention network for object detection and counting on drones, CoRR abs/1909.11307. arXiv:1909.11307.</br>
