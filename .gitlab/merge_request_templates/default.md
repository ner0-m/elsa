Thanks for the MR! If this is your first contribution to _elsa_ please read
[contribution guide](https://gitlab.lrz.de/IP/elsa/-/blob/master/CONTRIBUTING.md).


If you are still actively working on this, please mark this MR as draft (prefix the title it with
`Draft: `). If you are seeking help or feedback, tag the person you want help from. And to get them
merged, please add any/all of @lasser, @jj and/or @david.frank as Reviewers.

All of the things mentioned below are optional, but should provide you with some ideas of a good MR
and necessary steps to get your change merged. Some questions might not relevant for your MR, in
that case just delete them.

#### Short motivation of MR

- Why is this change necessary or useful? What can be done now, what couldn't be done prior?
- Link to issue, or other relevant discussion.

#### Short rational of implementation

- Why exactly this design? Did you investigate any other design? If so, why did you choose one
design over the other?

#### Checklist:

Some things you can think about, which can't be easily checked by CI:

- [ ] Does the code work and begets good code? If not, is there a different design which would
begets good code?
- [ ] Are the tests meaningful?
- [ ] Are the commits single-purpose, self-contained and follow the
[Angular commit format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)?
- [ ] Is the documentation updated, if it's necessary? Do you think a guide/tutorial would be
helpful/necessary for this feature?
