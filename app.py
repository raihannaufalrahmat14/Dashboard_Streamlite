import matplotlib.pyplot as plt

# Data for the donut chart (using the sentiment_counts DataFrame)
labels = sentiment_counts['Sentimen']
sizes = sentiment_counts['Percentage']
colors = ['#FF9999', '#66B2FF', '#99FF99'] # Custom colors for Negatif, Netral, Positif

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(8, 8))

# Create the pie chart
wedges, texts, autotexts = ax.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90,
                                    pctdistance=0.85, wedgeprops=dict(width=0.3, edgecolor='white'))

# Draw a circle in the center to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')

# Add a title
plt.title('Sentiment Distribution (Donut Plot)', fontsize=16)

# Create a legend with labels and percentages
legend_labels = [f'{l} ({s:.1f}%)' for l, s in zip(labels, sizes)]
ax.legend(wedges, legend_labels, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Improve text appearance
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(12)
for text in texts:
    text.set_fontsize(12)

plt.show()
