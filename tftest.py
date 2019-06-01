import os
import tensorflow as tf
import numpy as np


def main():
	x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
	#y_train = [0, 0, 0, 1]
	#y_train = tf.keras.utils.to_categorical(y_train, 2)
	y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])	# XOR

	x = tf.placeholder("float", [None, 2])
	initializer = tf.contrib.layers.variance_scaling_initializer()
	W1 = tf.Variable(initializer([2, 4]))
	b1 = tf.Variable(tf.zeros([4]))
	W2 = tf.Variable(initializer([4, 2]))
	b2 = tf.Variable(tf.zeros([2]))
	y1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
	y = tf.add(tf.matmul(y1, W2), b2)

	predict = tf.nn.softmax(y)

	y_ = tf.placeholder("float", [None, 2])

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)

	#opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)	# SGD
	opt = tf.train.AdamOptimizer(0.01).minimize(loss)		#Adam


	init = tf.global_variables_initializer()

	epochs = 20000
	with tf.Session() as sess:
		sess.run(init)
		for step in range(epochs):
			# バッチならここで選択
			sess.run(opt, feed_dict={x: x_train, y_: y_train})

			if step % (epochs/10) == 0:
				print(step/epochs*100, "%")
		
		for x_input in x_train:
			r = sess.run(predict, feed_dict={x: [x_input]})
			print(r)
			print(x_input, np.argmax(r, axis=1))

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	main()
