from ctypes import *

class StructHelper(object):
    def __get_value_str(self, name, fmt='{}'):
        val = getattr(self, name)
        if isinstance(val, Array):
            val = list(val)
        return fmt.format(val)

    def __str__(self):
        result = '{}:\n'.format(self.__class__.__name__)
        maxname = max(len(name) for name, type_ in self._fields_)
        for name, type_ in self._fields_:
            value = getattr(self, name)
            result += ' {name:<{width}}: {value}\n'.format(
                    name = name,
                    width = maxname,
                    value = self.__get_value_str(name),
                    )
        return result

    def __repr__(self):
        return '{name}({fields})'.format(
                name = self.__class__.__name__,
                fields = ', '.join(
                    '{}={}'.format(name, self.__get_value_str(name, '{!r}')) for name, _ in self._fields_)
                )

    @classmethod
    def _typeof(cls, field):
        """Get the type of a field
        Example: A._typeof(A.fld)
        Inspired by stackoverflow.com/a/6061483
        """
        for name, type_ in cls._fields_:
            if getattr(cls, name) is field:
                return type_
        raise KeyError

    @classmethod
    def read_from(cls, f):
        result = cls()
        if f.readinto(result) != sizeof(cls):
            raise EOFError
        return result

    def get_bytes(self):
        """Get raw byte string of this structure
        ctypes.Structure implements the buffer interface, so it can be used
        directly anywhere the buffer interface is implemented.
        https://stackoverflow.com/q/1825715
        """

        # Works for either Python2 or Python3
        return bytearray(self)

        # Python 3 only! Don't try this in Python2, where bytes() == str()
        #return bytes(self)

################################################################################

class CB_FILE(LittleEndianStructure, StructHelper):
    """
    Define a little-endian structure, and add our StructHelper mixin.
    C structure definition:
        __attribute__((packed))
        struct Vehicle
        {
            uint16_t    doors;
            uint32_t    price;
            uint32_t    miles;
            uint16_t    air_pressure[4];
            char        name[16];
        }
    """

    # Tell ctypes that this structure is "packed",
    # i.e. no padding is inserted between fields for alignment
    _pack_ = 1

    # Lay out the fields, in order
    _fields_ = [
        ('event_type',                  c_uint8),
        ('event_datetime',              c_uint32),
        ('event_millisec',              c_uint32),
        ('alert_level',                 c_uint8),
        ('contact_duty_A',              c_float),
        ('contact_duty_B',              c_float),
        ('contact_duty_C',              c_float),
        ('accum_contact_duty_A',        c_float),
        ('accum_contact_duty_B',        c_float),
        ('accum_contact_duty_C',        c_float),
        ('coil_integral_t1',            c_float),
        ('coil_max_current_t1',         c_float),
        ('coil_female_time_t1',         c_float),
        ('coil_integral_t2',            c_float),
        ('coil_max_current_t2',         c_float),
        ('coil_female_time_t2',         c_float),
        ('coil_integral_close',         c_float),
        ('coil_max_current_close',      c_float),
        ('coil_female_time_close',      c_float),
        ('contact_optime_A',            c_float),
        ('contact_optime_B',            c_float),
        ('block_close_time_A',          c_float),
        ('block_close_time_B',          c_float),
        ('block_close_time_C',          c_float),
        ('op_cnt',                      c_uint32),
        ('smp_per_cyc',                 c_uint16),
        ('cyc_count',                   c_uint16),
        ('trip1_coil_current',          c_uint16 * 2304),
        ('trip2_coil_current',          c_uint16 * 2304),
        ('close_coil_current',          c_uint16 * 2304),
        ('phase_current_A',             c_uint16 * 2304),
        ('phase_current_B',             c_uint16 * 2304),
        ('phase_current_C',             c_uint16 * 2304),
        ('initiate_and_contact',        c_uint8 * 2304)
    ]
