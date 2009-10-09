#!/usr/bin/env python

from Numeric import *
from RandomArray import uniform

class pyAnnError(Exception):
      def __init__(self,value):
            self.value=value
      def __str__(self):
            return repr(self.value)
      pass

def sigmoid(x):
      return tanh(x)
      return 1.0/(1.0+exp(x))

def dsigmoid(y):
      return 1.0-y**2
      return y*(1-y)

def dtanh(x):
      return 1.0*x-x*x

def diagonal_from_array(a):
      """
      build dianonal matrix from array:
        [1,3,4] => 
        [[1, 0, 0],
         [0, 3, 0], 
         [0, 0, 4]]
      """
      l=len(a)
      return [[i==j and a[i] or 0 for j in range(l) ] for i in range(l)]

def resize_using_defaults(a,newsize,default=1.0,rand=0):
      """
      resize a matrix to 'newsize', and fill all extra cells with 'default'
      """
      oldshape=shape(a)
      dif=array(newsize)-array(oldshape)
      if dif[0] <= 0 :
            tmpa=a[:newsize[0]]
      else:
            if rand==0:
                  _add_a=ones((dif[0],oldshape[1]))*default
            else:
                  _add_a=uniform(-1.0,1.0,(dif[0],oldshape[1]))*default
            tmpa=concatenate((a,_add_a))
      if dif[1] <= 0 :

            tmpb=array(tmpa)[:,:newsize[1]]
      else:
            if rand==0:
                  _add_b=ones((newsize[0],dif[1]))*default
            else:
                  _add_b=uniform(-1.0,1.0,(newsize[0],dif[1]))*default
            tmpb=concatenate((tmpa,_add_b),1)
      return tmpb
      

class pyAnn:
      def __init__(self, layers,strengths=[],default=1.0,rand=0):
            """
            layers: Array of neuron numbers in each layer,
              from input to output.
            strengths: the initial weigth matrixes, 
              will be automatically resized to fit with new dimension.
            default: the default value of weigths
            rand: rand=1 will initialize the weigths with random numers between -defaults and defaults
            
            """
            l2=map(lambda x:int(x), layers)
            l1=filter(lambda x:x>0, l2)
            
            if len(l1)<2:
                  raise pyAnnError("not enough layers")
            self.layers=l1
            self.nl=len(l1)
            self.neurons=[]
            self.__load_strength(strengths,default,rand)
            
      def __load_strength(self, strengths, default=1.0,rand=0):
            """
            strength: matrix of strengths between each layer,
            default: the default stength
            """
            import random
            self.strengthmatrixes=[]
            loadlen=len(strengths)
            for i in range(0, loadlen):
                  fill=default
                  self.strengthmatrixes.append(
                        resize_using_defaults(strengths[i], (self.layers[i],self.layers[i+1]),fill,rand)
                        )
            if loadlen < self.nl:
                  for i in range(loadlen,self.nl-1):
                        fill=default
                        if rand!=0:
                              self.strengthmatrixes.append(
                                    uniform(-1.0,1.0,(self.layers[i],self.layers[i+1]))*fill
                                    )
                        else:
                              self.strengthmatrixes.append(
                                    ones((self.layers[i],self.layers[i+1]))*fill
                                    )
      def __save_strength_to_file(self, xfile):
            for i in self.strengthmatrixes:
                  xfile.write(i.__repr__()+"\n")
      def __repr__(self):
            _t=""
            for i in self.strengthmatrixes:
                  _t=_t+repr(i) + "\n"
            for i in self.neurons:
                  _t=_t+repr(i) + "\n"
            return _t
      def __feed_forward(self, input,defualt=1.0, sgmd=sigmoid):
            self.neurons=[input]
            for i in self.strengthmatrixes:
                  _t=dot(self.neurons[-1], i)
                  self.neurons.append(sgmd(_t))

            return self.neurons[-1]

      def feed_forward(self, input, default=1.0):
            """
            the feed_forward of network, 
            returns the output neurons' value
            """
            if len(input)<self.layers[0]:
                  _input=list(input)+list(
                        ones(self.layers[0]-len(input))*default
                        )
            else:
                  _input=input
            return self.__feed_forward(_input, sgmd=sigmoid)

      	  
      def __bp(self, targets,sig=dsigmoid, fN=0.5,r=0.1):
            _t=targets
            ### calculate the delta of output layer
            _delta = map(lambda x,y: (x-y)*dsigmoid(y),
                         _t, self.neurons[-1])
            ###print "neurons", self.neurons, "+++"
            _sq_error = 0.0
            for ei in range(self.layers[-1]):
                  _sq_error += 0.5*(targets[ei]-self.neurons[-1][ei])**2

            for n in range(self.nl-1,0,-1):
                  
                  # calculate the strength change
                  _rs = r*uniform(-1.0,1.0,(self.layers[n-1],self.layers[n]))*_sq_error
                  _change = dot(transpose([self.neurons[n-1]]) ,[_delta] )+_rs

                  ### calculate the delta of hidden layers
                  #### first we calculate the error of hidden layers
                  _error_h = dot(_delta, transpose(self.strengthmatrixes[n-1]))

                  #### then we have the delta 
                  ####_delta = dot( _diagn2, _error_h)
                  _delta = map(lambda x,y: x*dsigmoid(y), _error_h, self.neurons[n-1])
                  
                  self.strengthmatrixes[n-1] = self.strengthmatrixes[n-1] + fN*_change

            return _sq_error
            pass
      def bp(self,target,default=1.0, fN=0.5, r=0):
            """
            the backPropgation process.
            fN: the trainning factor
            r: random factor
               add random (error*r*random(-1.0,1.0)) to every weigth update.
            """
            if len(target) < self.layers[-1]:
                  _target = list(target) + list( 
                        ones(self.layers[-1]-len(target))*default
                        )
            else:
                  _target=target
            return self.__bp(_target, fN=fN, r=r)
                  
def demo():
      cc=[
            uniform(-0.2,0.2,(3,3)),
            uniform(2.0,2.0,(3,1)),
            ]
      a=pyAnn([2,10,10,2],default=.2,rand=1)
      print a
      trainset=[
            ([.5,0.0],[0.5,0.0]),
            ([0.0,0.0],[0.0,0.0]),
            ([0.0,.5],[0.5,0.5]),
            ([.5,.5],[0.0,0.5]),
            ]
      turns=10000
      errth=0
      for i in xrange(turns):
            err=0.0
            for t in trainset:
                  a.feed_forward(t[0])
                  err += a.bp(t[1],fN=0.3,r=0.000)
            if err<errth:
                  print "in turn: ", i
                  break
            if i%100==1:
                  print "error: ", err
            
      print a

      for ts in trainset:
            print ts[0],a.feed_forward(ts[0])
      
if __name__ == "__main__":
      demo()
