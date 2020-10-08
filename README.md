# Heatmap Learner Convolutional Neural Network for Object Counting and Localization
Member: <a href="https://github.com/ekilic/"> Ersin KILIÇ </a> </br>
Supervisor: <a href="https://avesis.erciyes.edu.tr/ozturks/"> Serkan ÖZTÜRK </a> </br>

<h2>Description</h2>

This project aims to implement a simple and effective single-shot detector model to count and locate the objects in aerial images. The code and data will be made available after peer review.

<h2>Results</h2>

Experiments on the two car datasets (<a href="https://lafi.github.io/LPN/">PUCPR+</a> and <a href="https://lafi.github.io/LPN/">CARPK</a>) show the state-of-the-art counting and localizing performance of the proposed method compared with existing methods.

<table>
<tbody>
<tr>
<td>&nbsp;CARPK Dataset</td>
<td>&nbsp;MAE</td>
<td>RMSE&nbsp;</td>
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
<b><td>Ours (VGG-16)</td>
<td>2.12</td>
  <td>3.02</td></b>
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
