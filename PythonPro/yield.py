mylist = [x*x for x in range(3)]
for i in mylist :
	print(i)
		
mygenerator = (x*x for x in range(3))
for i in mygenerator :
	print("geterator ",i)

print("-----------0000--------")
for i in mygenerator :
	print("geterator_ ",i)
	
# 看起来除了把 [] 换成 () 外没什么不同。但是，你不可以再次使用 for i in mygenerator , 因为生成器只能被迭代一次：先计算出0，然后继续计算1，然后计算4，一个跟一个的…
	
# yield ,你必须要理解：当你调用这个函数的时候，函数内部的代码并不立马执行 ，这个函数只是返回一个生成器对象，这有点蹊跷不是吗	
def createGenerator():
	mylist = range(3)
	for i in mylist:
		yield i*i*i
		
print("------------------")
mygenerator = createGenerator()
print(mygenerator) # mygenerator is an object!
for i in mygenerator:
	print(i)
	
	
class Bank(): # let's create a bank, building ATMs
	crisis = False
	def create_atm(self) :
		while not self.crisis :
			yield "$100"
			
hsbc = Bank()
corner_street_atm = hsbc.create_atm()
print(corner_street_atm.__next__())
print(corner_street_atm.__next__())
print([corner_street_atm.__next__() for cash in range(5)])

hsbc.crisis = True
print(corner_street_atm.__next__())
wall_street_atm = hsbc.create_atm()
print(wall_street_atm.__next__())
hsbc.crisis = False
print(corner_street_atm.__next__())
brand_new_atm = hsbc.create_atm()

for cash in brand_new_atm :
	print(cash)
			