# AMAZON-KDD-Cup-23-MISX
Source code developed during the competition. 

[Amazon KDD Cup 23- Task I](https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge/problems/task-1-next-product-recommendation)

<h3>Task 1</h3>

<p>Task 1 aims to predict the next product that a customer is likely to engage with, given their session data and the attributes of each product. The test set for Task 1 comprises data from English, German, and Japanese locales. Participants are required to create a program that can predict the next product for each session in the test set.</p>

<p>To submit their predictions, participants should provide a single parquet file in which each row corresponds to a session in the test set. For each session, the participant should predict 100 product IDs (ASINs) that are most likely to be engaged with, based on historical engagements in the session. The product IDs should be stored in a list and are listed in decreasing order of confidence, with the most confident prediction at index 0 and least confident prediction at index 99.</p>

<p>For example, if product_25 is the most confident prediction for a session, product_100 is the second most confident prediction, and product_199 is the least confident prediction for the same session, the participant's submission should list product_25 first, product_100 next, a lot of other predictions in the middle, and product_199 last.</p>

<p>Input example:</p>

<table>
	<thead>
		<tr>
			<th>locale</th>
			<th>example_session</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>UK</td>
			<td>[product_1, product_2, product_3]</td>
		</tr>
		<tr>
			<td>DE</td>
			<td>[product_4, product_5]</td>
		</tr>
	</tbody>
</table>

<p>Output example:</p>

<table>
	<thead>
		<tr>
			<th>next_item</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>[product_25, product_100,…, product_199]</td>
		</tr>
		<tr>
			<td>[product_333, product_123,…, product_231]</td>
		</tr>
	</tbody>
</table>

<p>The evaluation metric for Task 1 is Mean Reciprocal Rank (MRR).</p>

<p>Mean Reciprocal Rank (MRR) is a metric used in information retrieval and recommendation systems to measure the effectiveness of a model in providing relevant results. MRR is computed with the following two steps: (1) calculate the reciprocal rank. The reciprocal rank is the inverse of the position at which the first relevant item appears in the list of recommendations. If no relevant item is found in the list, the reciprocal rank is considered 0. (2) average of the reciprocal ranks of the first relevant item for each session.</p>

<p>MRR@K=1N∑t∈T1Rank(t),</p>

<p>where Rank(t) is the rank of the ground truth on the top K result ranking list of test session t, and if there is no ground truth on the top K ranking list, then we would set 1Rank(t)=0. MRR values range from 0 to 1, with higher values indicating better performance. A perfect MRR score of 1 means that the model always places the first relevant item at the top of the recommendation list. An MRR score of 0 implies that no relevant items were found in the list of recommendations for any of the queries or users.</p>
