import matplotlib.pyplot as plt

# x-coordinates of left sides of bars 
left = [1, 2, 3, 4]
  
# heights of bars
height = [68, 85, 83, 90]
  
# labels for bars
tick_label = ['PCA', 'IM', 'SVM', 'NN']
  
# plotting a bar chart
plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['gray', 'red', 'green', 'blue'])
  
# naming the x-axis
plt.xlabel('x - axis')
# naming the y-axis
plt.ylabel('y - axis')
# plot title
plt.title('F1 score of Anomaly Detection using HDFS Data')
  
# function to show the plot
plt.show()

# -----------------------------------

# x-coordinates of left sides of bars 
left = [1, 2, 3, 4]
  
# heights of bars
height = [46, 86, 87, 90]
  
# labels for bars
tick_label = ['PCA', 'IM', 'SVM', 'NN']
  
# plotting a bar chart
plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['gray', 'red', 'green', 'blue'])
  
# naming the x-axis
plt.xlabel('x - axis')
# naming the y-axis
plt.ylabel('y - axis')
# plot title
plt.title('F1 score of Anomaly Detection using BGL Data')
  
# function to show the plot
plt.show()

# -----------------------------------
  
# x-coordinates of left sides of bars 
left = [1, 2, 3, 4]
  
# heights of bars
height = [66, 67, 82, 88]
  
# labels for bars
tick_label = ['PCA', 'IM', 'SVM', 'NN']
  
# plotting a bar chart
plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['gray', 'red', 'green', 'blue'])
  
# naming the x-axis
plt.xlabel('x - axis')
# naming the y-axis
plt.ylabel('y - axis')
# plot title
plt.title('F1 score of Anomaly Detection using Papertrail Data')
  
# function to show the plot
plt.show()