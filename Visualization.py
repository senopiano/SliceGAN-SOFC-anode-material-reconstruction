# Visualize the loss of D & G trend

plt.figure(figsize=(10 , 5))
plt.title('G & D loss during training process')
plt.plot(G_losses , label = 'G')
plt.plot(D_losses , label = 'D')
plt.xlabel('iteration number')
plt.ylabel('Loss')
plt.legend()
plt.show()