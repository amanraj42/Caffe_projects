import numpy as np
import sys
from scipy.misc import imread
from scipy.misc import imsave
#caffe setup 
caffe_root='../'
sys.path.insert(0,caffe_root+'python')
sys.path.append('/home/wschool25/caffe-master/python')
import caffe
MODEL_FILE='/home/wschool25/caffe-master/examples/bubble/deploy.prototxt'
PRETRAINED='/home/wschool25/caffe-master/examples/bubble_iter_150000.caffemodel'
net=caffe.Classifier(MODEL_FILE,PRETRAINED,image_dims=(16,16))
net.set_phase_test()
net.set_mode_cpu()

fsize = 16
dS = 16
#p=1

imagefiles = ['test.jpg']
def rgb2gray(rgb):
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

for filename in imagefiles:
    im = imread(filename)
    im = rgb2gray(im)
    m,n = im.shape

    imsave('grayscale.jpeg',im)
    V = np.zeros((1,fsize*fsize))
    v1= np.zeros((16,16))
    for i in xrange(0,m,dS):
        for j in xrange(0,n,dS):
            ik = i + fsize
            jk = j + fsize
            if ((ik <= m) and (jk <= n)):
                k=0
                for a in xrange(i,ik):
                    for b in xrange(j,jk):
                       # print a,b
                        V[0][k] = im[a][b]
                        k=k+1
        	 		        
            v1=np.reshape(V,(16,16,1))
   	    print 'shape',v1.shape
   	    prediction=net.predict([v1])
            print 'prediction shape:',prediction[0].shape
   	    print 'prediction class:',prediction[0].argmax()
   	    Cl=prediction[0].argmax()
	    print 'class',Cl
   	#imsave('patch' + str(p) + '.jpeg',v1)
       #p=p+1

	    if ((ik <= m) and (jk <= n)):
   	    	if(Cl==1):
    		   for a in xrange(i,ik):
   		     for b in xrange(j,jk):
   			 im[a][b]=0
			 #print 'hi'
   
   	    	if(Cl==0):
   		   for a in xrange(i,ik):
   		     for b in xrange(j,jk):
   			 im[a][b]=255
			 #print 'ho'
   
    imsave('new.jpg',im)


                         


                    

