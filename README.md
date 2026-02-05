# Green Button

[![GitHub Release](https://img.shields.io/github/release/rhounsell/home-assistant-green-button.svg?style=for-the-badge)](https://github.com/rhounsell/home-assistant-green-button/releases)
[![GitHub Activity](https://img.shields.io/github/commit-activity/y/rhounsell/home-assistant-green-button?style=for-the-badge)](https://github.com/rhounsell/home-assistant-green-button/commits)
[![License][license-shield]](LICENSE)

[![pre-commit][pre-commit-shield]][pre-commit]
[![Black][black-shield]][black]

[![hacs][hacsbadge]][hacs]
[![Project Maintenance][maintenance-shield]][user_profile]

A custom component for Home Assistant that will import Green Button Usage and Cost data, which can then be added to the Energy dashboard.

The Green Button data needs to be in the ESPI XML Schema Definition, contained in an Atom Syndication format. 

This custom component has been developed to handle the Green Button data available from Hydro Ottawa and Enbridge Gas. It may or may not work with other sources of Green Button data.

## Installation (HACS not verified yet)

1. Copy the green_button folder under custom_components into your Home Assistant custom_components folder
2. Restart Home Assistant
3. In the HA UI go to "Configuration" -> "Integrations" click "+" and search for "Green Button"
4. Follow the [Energy Dashboard Setup](ENERGY_DASHBOARD_SETUP.md) guide to add sensors to your Energy Dashboard

**Note**: Sensors will show "unknown" state - this is intentional and doesn't affect functionality. See [RECORDER_CONFIG.md](RECORDER_CONFIG.md) for details.

## Credits

This project was generated from [@oncleben31](https://github.com/oncleben31)'s [Home Assistant Custom Component Cookiecutter](https://github.com/oncleben31/cookiecutter-homeassistant-custom-component) template.

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
