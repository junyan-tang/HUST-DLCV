z = torch.randn(2, latent_dim)
a = torch.FloatTensor(latent_dim, 10)
for i in range(latent_dim):
    a[i] = torch.linspace(z[0][i], z[1][i], 10)

b = a.t()
gen_images = generator(b)
torchvision.utils.save_image(gen_images.data[:], "images_trans.png", normalize=True, nrow=10)
