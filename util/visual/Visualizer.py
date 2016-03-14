import matplotlib.pyplot as plt

import numpy as np


class Visualizer(object):



    @staticmethod
    def plot_vector(vectorz, v_color='blue'):


        # the histogram of the data
        mp = plt.subplot()
        step = 1.0/len(vectorz)
        for i in np.arange(len(vectorz)):
            v = vectorz[i]
            mp.bar(np.arange(len(v[0])) + step*i,v[0], width=step,color=v[1])

        plt.xlabel('dimensions')
        plt.ylabel('values')
        plt.grid(True)

        plt.show()
        """trace = go.Histogram(
            x=vector,
            histnorm='count',
            name='control',
            autobinx=False,
            xbins=dict(
                start=-3.2,
                end=2.8,
                size=0.2
            ),
            marker=dict(
                color= v_color,
                line=dict(
                    color='grey',
                    width=0
                )
            ),
            opacity=0.75
        )

        data = [trace]
        layout = go.Layout(
            title='Sampled Results',
            xaxis=dict(
                title='Value'
            ),
            yaxis=dict(
                title='Count'
            ),
            barmode='overlay',
            bargap=0.25,
            bargroupgap=0.3
        )
        fig = go.Figure(data=data, layout=layout)
        plot_url = py.plot(fig, filename='style-histogram')"""



if __name__ == '__main__':
    Visualizer.plot_vector([1,2,3,4,3,2,1])
