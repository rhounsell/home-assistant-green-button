# Green Button

[![GitHub Release](https://img.shields.io/github/release/rhounsell/home-assistant-green-button.svg?style=for-the-badge)](https://github.com/rhounsell/home-assistant-green-button/releases)
[![GitHub Activity](https://img.shields.io/github/commit-activity/y/rhounsell/home-assistant-green-button?style=for-the-badge)](https://github.com/rhounsell/home-assistant-green-button/commits)
[![License][license-shield]](LICENSE)

[![pre-commit][pre-commit-shield]][pre-commit]
[![Black][black-shield]][black]

[![hacs][hacsbadge]][hacs]
[![Project Maintenance][maintenance-shield]][user_profile]

A custom component for Home Assistant that will import Green Button Usage and Cost data, and then generate statistics which can be added to the Energy dashboard.

The Green Button data needs to be in the ESPI XML Schema Definition, contained in an Atom Syndication format. 

This custom component has been developed to handle the Green Button data available from Hydro Ottawa and Enbridge Gas. It may or may not work with other sources of Green Button data.

## Installation (HACS not set up yet)

1. Copy the green_button folder under custom_components into your Home Assistant custom_components folder
2. Restart Home Assistant
3. In the HA UI go to "Configuration" -> "Integrations". Click "+" and search for "Green Button"
4. Complete the installation with or without providing Green Button XML data
   - If you skip the XML import during setup, you can import it later using the **Add Entry** button on the Green Button integration or via **Developer Tools → Actions → 'Import Green Button ESPI XML'**

By default, importing electricity usage and billing data will create a "Home Electricity" Green Button device, with entities named "sensor.home_electricity_cost" and "sensor.home_electricity_usage". Importing Natural Gas data will create by default a "Home Natural Gas" device, with "sensor.home_natural_gas_cost" and "sensor.home_natural_gas_usage" sensors.

Statistics will automatically be generated for these sensors and can be imported into the Energy dashboard. The "usage" sensors have a state_class of "total_increasing", and the "cost" sensors have a state_class of "total". It's the statistics that you'll want to examine for periodic usage rather than the raw data in the sensors themselves.

It may take a few minutes for all the associated statisitics to be generated. The related sensor may not be available to add into the Energy dashbord until the generation is completed.

Review the [Green Button Component Description](GREEN_BUTTON_COMPONENT_DESCRIPTION.md) for detail on how the Green Button custom component functions.

## Data Imports

Green Button XML blocks do not need to be imported in chronological order, and can have gaps between the dates for imported blocks of data. Any data that is imported that overlaps pre-existing data will be merged with that data. If there is overlapping data, the statistics generation will resolve them. The drawback to overlapping data will be a larger than necessary green_button_xml file in /config/.storage

When importing xml files, it's probably easiest to locate the file to be imported in Home Assistant's **share** folder, which appears under the config folder, and can be referenced in the file path using **/share/**/<green_button_xml_file.xml>

## Services/Actions

There are several actions (services) related to the Green Button custom component available under **Developer Tools → Actions**. As of this writing, they are:
- Log Green Button Meter Reading Intervals
- Log Stored Green Button XML Info
- Delete Green Button Statistics
- Import Green Button ESPI XML
- Clear Stored Green Button XML Data

## Notes

None of the original tests or development support files such as .pre-commit-config.yaml have been updated or, for that matter, used when updating this component.

## Credits

This project was originally generated from [@oncleben31](https://github.com/oncleben31)'s [Home Assistant Custom Component Cookiecutter](https://github.com/oncleben31/cookiecutter-homeassistant-custom-component) template.

Code template was mainly taken from [@Ludeeus](https://github.com/ludeeus)'s [integration_blueprint][integration_blueprint] template.

Forked from the Green Button project created by [@vqvu](https://github.com/vqvu).

---

[integration_blueprint]: https://github.com/custom-components/integration_blueprint
[black]: https://github.com/psf/black
[black-shield]: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
[commits-shield]: https://img.shields.io/github/commit-activity/y/vqvu/home-assistant-green-button.svg?style=for-the-badge
[commits]: https://github.com/vqvu/home-assistant-green-button/commits/main
[hacs]: https://hacs.xyz
[hacsbadge]: https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge
[license-shield]: https://img.shields.io/github/license/vqvu/home-assistant-green-button.svg?style=for-the-badge
[maintenance-shield]: https://img.shields.io/badge/maintainer-%40rhounsell-blue.svg?style=for-the-badge
[pre-commit]: https://github.com/pre-commit/pre-commit
[pre-commit-shield]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=for-the-badge
[releases-shield]: https://img.shields.io/github/release/vqvu/home-assistant-green-button.svg?style=for-the-badge
[releases]: https://github.com/vqvu/home-assistant-green-button/releases
[user_profile]: https://github.com/rhounsell
