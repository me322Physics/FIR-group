def recur_fibo(n):
   """Recursive function to
   print Fibonacci sequence"""
   if n <= 1:
       return n
   else:
       return(recur_fibo(n-1) + recur_fibo(n-2))

# Change this value for a different result
nterms = 10

# uncomment to take input from the user
#nterms = int(input("How many terms? "))
print('started running python script')
# check if the number of terms is valid
print("Fibonacci sequence:")
for i in range(nterms):
       print(recur_fibo(i))

