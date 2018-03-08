
# coding: utf-8

# In[9]:


people = ['Dr. Christopher Brooks', 'Mr. Habeebur Rahman', 'Ms. Aarthi Thirumavalavan']

def split_title_name(x):
    title = x.split(' ')[0]
    lastname = x.split(' ')[-1]
    return '{} {}'.format(title, lastname)

list(map(split_title_name, people))
    
    


# In[10]:


people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
    title = person.split()[0]
    lastname = person.split()[-1]
    return '{} {}'.format(title, lastname)

A = list(map(split_title_and_name, people))


for item in A:
	print(item)
