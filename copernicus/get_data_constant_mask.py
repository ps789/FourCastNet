import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'land_sea_mask', 'soil_type', 'geopotential'
        ],
        'year': ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'],
        'month':  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'day': [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'
        ],
        'time': [
            '00:00'
        ],
        'grid':['2.5', '2.5']

    },
    '../ERA5_data/data_constant_mask.nc')


