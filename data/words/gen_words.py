WORDS = \
'''a
about
all
also
and
as
at
be
because
but
by
can
come
could
day
do
even
find
first
for
from
get
give
go
have
he
her
here
him
his
how
I
if
in
into
it
its
just
know
like
look
make
man
many
me
more
my
new
no
not
now
of
on
one
only
or
other
our
out
people
say
see
she
so
some
take
tell
than
that
the
their
them
then
there
these
they
thing
think
this
those
time
to
two
up
use
very
want
way
we
well
what
when
which
who
will
with
would
year
you
your'''.split('\n')


with open("words.csv", "w") as f:
    for w in WORDS:
        # first letter
        f.write(f'word "{w}" starts with "{w[0]}"\n')
        f.write(f'"{w}" begins with "{w[0]}"\n')
        f.write(f'the first letter in "{w}" is "{w[0]}"\n')
        # last letter
        f.write(f'word "{w}" ends with "{w[-1]}"\n')
        f.write(f'"{w}" ends with "{w[-1]}"\n')
        f.write(f'the last letter in "{w}" is "{w[-1]}"\n')

