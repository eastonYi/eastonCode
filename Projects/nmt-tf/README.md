# standards

## projects
different projets has different dirs. For models in the the same project can be compare, there is a one-to-one correspondence between a project and the data it uses.
Different projects could have different 'remote sync', providing the `.remote-sync.json` file in the folder and opening it as a new project in the Atom.

So, if you are working among various dictionaries, open the `asf-tf` as one projects, and different exps has their individual projects.


## data
the dataset is the raw feature sequence if transforme==False and that is the dataset for the savetfdata. we set train and dev dataset with transforme=False, which means store the data with raw form. We transform the feature sequence during the tf graph after read the data from tfdata. During the infer, which is the test dataset, we set it to transform=True
