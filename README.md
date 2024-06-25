# m3ip-homework
homework of the course m3ip

## checked 11-20 (to check 17: too slow in the last part, maybe IRLS broken?, do 18)

### Main code of 18:
```python
for i in range(0, imsz[0]):
    for j in range(0, imsz[1]):
        # extract the patch from img_pad whose center has the location (i, j) in the noisy image
        s = img_pad[i:i+p, j:j+p]

        # initialize the pixel estimate and the total weights
        pixel_hat = 0
        weight = 0
        for r in range(max(0, i - H), min(imsz[0] - p // 2, i + H)):
            for c in range(max(0, j - H), min(imsz[1] - p // 2, j + H)):
                # extract the patch
                z = img_pad[r:r+p, c:c+p]

                # compute the distance with the reference patch
                d = np.linalg.norm(s - z)

                w = np.exp(-d / (M * sigma_noise ** 2))

                # update the weight and the pixel estimate
                pixel_hat += w * img_pad[r + p // 2, c + p // 2]
                weight += w

        # estimate the pixel (i, j) as the weighted average of the central pixel of the extracted patches
        img_hat[i, j] = pixel_hat / weight
```

## to check 1-10 (in particular 10 (KSVD))