#Release scripts

## What does it do

The script will create a new release in GitLab with a given name, tag and description.
All the releases can be seen at[elsa releases](https://gitlab.lrz.de/IP/elsa/-/releases).

## How to run the script

#### #Requierments

You 'll need Python to run the script (most likely 3, I' ve not tested 2). Further the package `python-gitlab` 
will be needed.Install it with `pip install python- gitlab`.

You'll also need an access token for GitLab. 
[Refer to this](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html#creating-a-personal-access-token).

#### Arguments

The basic usage would be something along the line to :

```bash
python release.py --private-token "YOUR_PRIVATE_TOKEN" --name "New Release" --tag "v0.7.0" --summary-file summary.md
```

You can further customize other things,see `python release.py --help` to see all the possibilities.
