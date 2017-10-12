from dedalus.core.operators import *

@parseable
@addname('part_integ')
def integrate(arg0, bases, starts, ends, out=None):
    # Cast to operand
    arg0 = Operand.cast(arg0)
    # No bases: integrate over whole domain
    if len(bases) == 0:
        bases = arg0.domain.bases
    # Multiple bases: apply recursively
    if len(bases) > 1:
        arg0 = integrate(arg0, *bases[:-1], *starts[:-1], *ends[:-1])
    # Call with single basis
    basis = arg0.domain.get_basis_object(bases[-1])
    start = starts[-1]
    end   = ends[-1]
    f     = arg0.domain.new_field()
    f_int = arg0.domain.new_field()
    f.set_scales(arg0.domain.dealias, keep_data=False)
    f['g'] = arg0.evaluate().data
    f.antidifferentiate(basis, ('left', 0), out=f_int)
    f_int.interpolate(**{str(basis): float(start)}, out=f)
    f['g'] -= f_int.interpolate(**{str(basis):float(end)})['g']
    return f


