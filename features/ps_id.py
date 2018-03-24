
class PSID(object):

    def generate_id(self, data):

        columns = data.columns

        columns_selected = [column for column in columns[columns.str.endswith('_cat')]
                            if len(data[column].unique()) > 3 and data[column].min() > -1]

        ids = []

        for index, row in data.iterrows():
            current_id = (row[columns_selected[2]] * 1000000 + row[columns_selected[0]] * 1000 + row[columns_selected[1]])
            ids.append(current_id)

            if index % 10000 == 0:
                print('Converting {} of {}'.format(index, data.shape[0]))

        return ids