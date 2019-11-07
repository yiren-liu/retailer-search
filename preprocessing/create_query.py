#
# with open('../data/search_query.csv','w') as f:
#     for num in range(100):
#         description=input('description:')
#         description=description.replace('\t',' ')
#         querys=''
#         for query_num in range(2):
#             query=input('query:')
#             querys=querys+'\t'+query
#         f.write(str(num)+'\t'+description+'\t'+querys+'\n')


def remove_dup():
    with open('../data/search_query.csv', 'r') as f:
        lines=f.readlines()

    lines=set(lines)
    with open('../data/search_query.csv', 'w') as f:
        for line in lines:
            f.write(line)



if __name__=='__main__':
    remove_dup()