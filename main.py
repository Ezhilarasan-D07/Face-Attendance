from systools import reduce
def counter(a,b):  
    return a+b
#for i in array:
#flag+=i
#return flag 

array=[2,1,3,4,5]

#(counter(array))
#2+1+3+4+5=--> 15

result = reduce(counter(a,b),array)
print(result)
