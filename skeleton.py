import tensorflow as tf
import pickle
import gzip
import csv
from scipy import misc
import numpy as np
import cv2
import os, os.path
np.set_printoptions(threshold='nan')

class SVHN(object):

    path = "/data/train_images/"
    
    def __init__(self, data_dir):
        """
            data_directory : path like /home/rajat/mlproj/dataset/
                            includes the dataset folder with '/'
            Initialize all your variables here
        """
        self.data_directory = data_dir

        self.train_len = 33000
        self.track_digits = 0
        self.track_digits_position = 1
        self.track_images = 0
        self.batch_size = 75

        self.sess = tf.InteractiveSession()

        # initializing input images, input lengths and length of sequence
        self.x = tf.placeholder(tf.float32, shape = [None,4096,3])
        self.y_digit_length = tf.placeholder(tf.float32, shape = [None,10])
        self.y_digit1 = tf.placeholder(tf.float32, shape = [None,10])
        self.y_digit2 = tf.placeholder(tf.float32, shape = [None,10])
        self.y_digit3 = tf.placeholder(tf.float32, shape = [None,10])
        self.y_digit4 = tf.placeholder(tf.float32, shape = [None,10])
        self.y_digit5 = tf.placeholder(tf.float32, shape = [None,10])


        # x_temp = np.zeros([10,4096,3],dtype=float,order='C')

        # initializing weights and biases and model
            # first conv layer
        self.W_conv1 = self.weight_variable([5,5,3,32])
        self.b_conv1 = self.bias_variable([32])
        self.x_image = tf.reshape(self.x, [-1,64,64,3])
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pol_2x2(self.h_conv1)

            # second conv layer
        self.W_conv2 = self.weight_variable([5,5,32,64])
        self.b_conv2 = self.bias_variable([64])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pol_2x2(self.h_conv2)
            # printing shape of the image
        # print self.h_pool2.shape

            # third conv layer
        self.W_conv3 = self.weight_variable([5,5,64,128])
        self.b_conv3 = self.bias_variable([128])
        self.h_conv3 = tf.nn.relu(self.conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)
        self.h_pool3 = self.max_pol_2x2(self.h_conv3)
        # print self.h_pool3.shape

            # fourth conv layer
        self.W_conv4 = self.weight_variable([5,5,128,200])
        self.b_conv4 = self.bias_variable([200])
        self.h_conv4 = tf.nn.relu(self.conv2d(self.h_pool3, self.W_conv4)+self.b_conv4)
        self.h_pool4 = self.max_pol_2x2(self.h_conv4)
        # print self.h_pool4.shape

            # fifth conv layer
        self.W_conv5 = self.weight_variable([3,3,200,300])
        self.b_conv5 = self.bias_variable([300])
        self.h_conv5 = tf.nn.relu(self.conv2d(self.h_pool4, self.W_conv5)+self.b_conv5)
        self.h_pool5 = self.max_pol_2x2(self.h_conv5)
        print self.h_pool5.shape

            # densely connected layer shape of the image at this point is 16x16
        self.W_fc1 = self.weight_variable([2*2*300, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool5, [-1, 2*2*300])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        # print self.h_fc1.shape

            # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

            # readout layer and output digits
        self.W_fc2_len = self.weight_variable([1024, 10])
        self.b_fc2_len = self.bias_variable([10])

        self.W_fc2_1 = self.weight_variable([1024, 10])
        self.b_fc2_1 = self.bias_variable([10])

        self.W_fc2_2 = self.weight_variable([1024, 10])
        self.b_fc2_2 = self.bias_variable([10])

        self.W_fc2_3 = self.weight_variable([1024, 10])
        self.b_fc2_3 = self.bias_variable([10])

        self.W_fc2_4 = self.weight_variable([1024, 10])
        self.b_fc2_4 = self.bias_variable([10])

        self.W_fc2_5 = self.weight_variable([1024, 10])
        self.b_fc2_5 = self.bias_variable([10])

        self.y_pred_digit_length = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2_len) + self.b_fc2_len)
        self.y_pred_digit1 = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2_1) + self.b_fc2_1)
        self.y_pred_digit2 = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2_2) + self.b_fc2_2)
        self.y_pred_digit3 = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2_3) + self.b_fc2_3)
        self.y_pred_digit4 = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2_4) + self.b_fc2_4)
        self.y_pred_digit5 = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2_5) + self.b_fc2_5)

        self.cross_entropy_len = tf.reduce_mean(-tf.reduce_sum(self.y_digit_length*tf.log(tf.clip_by_value(self.y_pred_digit_length,1e-10,1.0)), reduction_indices=[1]))
        self.cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(self.y_digit1*tf.log(tf.clip_by_value(self.y_pred_digit1,1e-10,1.0)), reduction_indices=[1]))
        self.cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(self.y_digit2*tf.log(tf.clip_by_value(self.y_pred_digit2,1e-10,1.0)), reduction_indices=[1]))
        self.cross_entropy3 = tf.reduce_mean(-tf.reduce_sum(self.y_digit3*tf.log(tf.clip_by_value(self.y_pred_digit3,1e-10,1.0)), reduction_indices=[1]))
        self.cross_entropy4 = tf.reduce_mean(-tf.reduce_sum(self.y_digit4*tf.log(tf.clip_by_value(self.y_pred_digit4,1e-10,1.0)), reduction_indices=[1]))
        self.cross_entropy5 = tf.reduce_mean(-tf.reduce_sum(self.y_digit5*tf.log(tf.clip_by_value(self.y_pred_digit5,1e-10,1.0)), reduction_indices=[1]))

        self.final_entropy = self.cross_entropy1+self.cross_entropy2+self.cross_entropy3+self.cross_entropy4+ self.cross_entropy5 + self.cross_entropy_len
        tf.summary.scalar('final entropy', self.final_entropy)

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.final_entropy)

        self.correct_prediction_len = tf.equal(tf.argmax(self.y_pred_digit_length,1), tf.argmax(self.y_digit_length,1))
        self.correct_prediction1 = tf.equal(tf.argmax(self.y_pred_digit1,1),tf.argmax(self.y_digit1,1))
        self.correct_prediction2 = tf.equal(tf.argmax(self.y_pred_digit2,1),tf.argmax(self.y_digit2,1))
        self.correct_prediction3 = tf.equal(tf.argmax(self.y_pred_digit3,1),tf.argmax(self.y_digit3,1))
        self.correct_prediction4 = tf.equal(tf.argmax(self.y_pred_digit4,1),tf.argmax(self.y_digit4,1))
        self.correct_prediction5 = tf.equal(tf.argmax(self.y_pred_digit5,1),tf.argmax(self.y_digit5,1))
        # print self.correct_prediction5.shape
        # determining accuracy
        self.accuracy_len = tf.reduce_mean(tf.cast(self.correct_prediction_len,tf.float32))
        self.accuracy_digit1 = tf.reduce_mean(tf.cast(self.correct_prediction1,tf.float32))
        self.accuracy_digit2 = tf.reduce_mean(tf.cast(self.correct_prediction2,tf.float32))
        self.accuracy_digit3 = tf.reduce_mean(tf.cast(self.correct_prediction3,tf.float32))
        self.accuracy_digit4 = tf.reduce_mean(tf.cast(self.correct_prediction4,tf.float32))
        self.accuracy_digit5 = tf.reduce_mean(tf.cast(self.correct_prediction5,tf.float32))

        self.out1 = tf.argmax(self.y_pred_digit_length,1)
        self.out2 = tf.argmax(self.y_pred_digit1,1)
        self.out3 = tf.argmax(self.y_pred_digit2,1)
        self.out4 = tf.argmax(self.y_pred_digit3,1)
        self.out5 = tf.argmax(self.y_pred_digit4,1)
        self.out6 = tf.argmax(self.y_pred_digit5,1)

        merged = tf.summary.merge_all()
        test_writer = tf.summary(FileWriter(FLAGS.summaries_dir+'/train'))



    def train(self):
        

        self.sess.run(tf.global_variables_initializer())
        for i in range (1,5):
            self.temp_x = self.getImages()
            self.temp_y_digit_length,self.temp_y_digit1,self.temp_y_digit2,self.temp_y_digit3,self.temp_y_digit4,self.temp_y_digit5 = self.getDigits()
            q,a,b,c,d,e,f,g = self.sess.run([merged,self.train_step, self.final_entropy,self.cross_entropy1,self.cross_entropy2,
                self.cross_entropy3,self.cross_entropy4,self.cross_entropy_len], feed_dict={
                self.x: self.temp_x.astype(np.float32),
                self.y_digit_length: self.temp_y_digit_length.astype(np.float32),
                self.y_digit1: self.temp_y_digit1.astype(np.float32), self.y_digit2: self.temp_y_digit2.astype(np.float32),
                self.y_digit3: self.temp_y_digit3.astype(np.float32), self.y_digit4: self.temp_y_digit4.astype(np.float32),
                self.y_digit5: self.temp_y_digit5.astype(np.float32),
                self.keep_prob: 0.5})


    def getDigits(self):
        leng = np.zeros([self.batch_size,10])
        d1 = np.zeros([self.batch_size,10])
        d2 = np.zeros([self.batch_size,10])
        d3 = np.zeros([self.batch_size,10])
        d4 = np.zeros([self.batch_size,10])
        d5 = np.zeros([self.batch_size,10])

        with open (self.data_directory+'train.csv', 'rb') as f:
            reader = csv.reader(f,delimiter = ',')
            reader = list(reader)
            temp_len = 0;
            num_images=0;
            i = 1
            while(num_images<self.batch_size):
                row = reader[self.track_digits_position]
                # print 'outside while'+ str(self.track_digits_position)
                temp1 = row[0]
                temp_len = temp_len + 1
                d1[num_images][int(row[1])%10] = 1
                # getting corners of image
                # print row
                while (1):

                    self.track_digits_position = self.track_digits_position+1
                    row = reader[self.track_digits_position]
                    if (row[0]==temp1):
                        # print 'inside while '+str(self.track_digits_position)
                        temp_len = temp_len + 1
                        if (temp_len==2):
                            d2[num_images][int(row[1])%10] = 1
                        elif (temp_len==3):
                            d3[num_images][int(row[1])%10] = 1
                        elif (temp_len==4):
                            d4[num_images][int(row[1])%10] = 1
                        else:
                            d5[num_images][int(row[1])%10] = 1

                    else:
                        leng[num_images][temp_len] = 1

                        num_images = num_images + 1
                        # i = i-1
                        self.track_digits_position = self.track_digits_position - 1
                        temp_len = 0
                        break

                if (num_images>=self.batch_size):
                    break
                else:
                    self.track_digits_position = self.track_digits_position+1
        self.track_digits_position = self.track_digits_position+1
        if (self.track_images == 0):
            self.track_digits_position = 1

        return leng,d1,d2,d3,d4,d5

    def getImages(self):
        inp = np.zeros([self.batch_size,4096,3])
        self.track_images = self.track_images%self.train_len
        i=0
        while (1):
            img = cv2.imread(self.data_directory+'cr/'+str(self.track_images+1)+'.png')
            img = img.reshape((4096,3))
            # print 'read image '+ str(self.track_images+1)
            inp[i] = img
            self.track_images = self.track_images+1
            i = i+1
            if (i>=self.batch_size):
                break
        self.track_images = (self.track_images)%self.train_len
        return inp

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

    def max_pol_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def get_sequence(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array

            return: list of integers with the sequence of digits. Example: [5,0,3] for an image having 503 as the sequence.

        """
        res = []
        image = cv2.resize(image,(64,64))
        kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(image, -1, kernel_sharpen_1)
        temp = np.zeros([1,4096,3])
        image = img.reshape([4096,3])
        temp[0] = image
        y,y1,y2,y3,y4,y5 = self.sess.run([self.out1, self.out2,
            self.out3,self.out4,self.out5,self.out6],
            feed_dict={self.x: temp.astype(np.float32), self.keep_prob: 1.0
            })
        if (y[0]>0):
            res.append(y1[0])
        if (y[0]>1):
            res.append(y2[0])
        if (y[0]>2):
            res.append(y3[0])
        if (y[0]>3):
            res.append(y4[0])
        if (y[0]>4):
            res.append(y5[0])

        return res


    def preprocess(self):
        # preprocess images
        kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        with open (self.data_directory+'train.csv', 'rb') as f:
            reader = csv.reader(f,delimiter = ',')
            reader = list(reader)
            i = 1
            while(i<len(reader)):
                row = reader[i]
                temp1 = row[0]
                # getting corners of image
                x_start = abs(int(row[2]))
                x_finish = x_start + abs(int(row[4]))
                y_top = abs(int(row[3]))
                y_bottom = y_top + abs(int(row[5]))
                print row
                while (1):
                    if (i==(len(reader)-1)) :
                        break

                    else:
                        i = i+1
                        row = reader[i]
                        if (row[0]==temp1):

                            if (x_start > abs(int(row[2]))):
                                x_start = abs(int(row[2]))

                            x_finish = max(x_finish, abs(int(row[2])) + abs(int(row[4])))
                            
                            if (y_bottom < (abs(int(row[3]))+abs(int(row[5])))):
                                y_bottom = (abs(int(row[3])) + abs(int(row[5])))
                            
                            if (y_top > abs(int(row[3]))):
                                y_top = abs(int(row[3]))

                        else:
                            i = i-1
                            row = reader[i]
                            img = cv2.imread(self.data_directory+'train_images/'+row[0])
                            
                                # print x_start, x_finish, y_top, y_bottom
                            img = img[y_top:y_bottom, x_start:x_finish]
                            img = cv2.resize(img,(64,64))
                            # cv2.imshow('normal',img)
                            img = cv2.filter2D(img, -1, kernel_sharpen_1)
                            cv2.imwrite(self.data_directory+'cr/'+row[0],img)
                            break
                if (i==(len(reader)-1)):
                    row = reader[i]
                    img = cv2.imread(self.data_directory+'train_images/'+row[0])
                    
                    # print x_start, x_finish, y_top, y_bottom
                    img = img[y_top:y_bottom, x_start:x_finish]
                    img = cv2.resize(img,(64,64))
                    # cv2.imshow('normal',img)
                    img = cv2.filter2D(img, -1, kernel_sharpen_1)
                    cv2.imwrite(self.data_directory+'cr/'+row[0],img)
                    break
                else:
                    print x_start, x_finish, y_top, y_bottom
                    i = i+1



    def save_model(self):

        self.saver = tf.train.Saver()
        self.save_path = self.saver.save(self.sess, "./model.ckpt")

        print self.save_path

        """
            saves model on the disk

            no return expected
        """


    @staticmethod
    def load_model(**params):

        svhn = SVHN(params['name'])

        svhn.saver = tf.train.Saver()
        svhn.sess = tf.InteractiveSession()

        svhn.saver.restore(svhn.sess, "./model.ckpt")
        print 'model loaded successfully'

        return svhn

        """
            returns a pre-trained instance of SVHN class
        """

if __name__ == "__main__":

        # obj = SVHN('dataset/')
        # obj.train()
        # obj.save_model(name="svhn.gz")
    obj = SVHN('data/')
    obj.train()
