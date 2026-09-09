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
2. If Green Button XML files will be imported from outside Home Assistant's `/config` directory, add each parent directory that will contain imports to the `homeassistant` section of `configuration.yaml`. For example, to import files from `/share`:

   ```yaml
   homeassistant:
     allowlist_external_dirs:
       - /share
   ```

   The `/config` directory is available by default; directories outside it must be allowlisted before the import can read files from them.
3. Restart Home Assistant
4. In the HA UI go to "Configuration" -> "Integrations". Click "+" and search for "Green Button"
5. Complete the installation with or without providing Green Button XML data
   - If you skip the XML import during setup, you can import it later using the **Add Entry** button on the Green Button integration or via **Developer Tools → Actions → 'Import Green Button ESPI XML'**

By default, importing electricity usage and billing data will create a "Home Electricity" Green Button device, with entities named "sensor.home_electricity_cost" and "sensor.home_electricity_usage". Importing Natural Gas data will create by default a "Home Natural Gas" device, with "sensor.home_natural_gas_cost" and "sensor.home_natural_gas_usage" sensors.

Statistics are automatically generated for these sensors and can be added to the Energy dashboard. The "usage" sensors have a state_class of "total_increasing", and the "cost" sensors have a state_class of "total". Examine the statistics, rather than the raw sensor state, for periodic usage.

Imported Green Button history is stored in integration-owned statistics with stable IDs. This keeps imported history intact if a display entity is renamed or recreated, and prevents Home Assistant's automatic sensor-statistics pipeline from duplicating it.

It may take a few minutes for all associated statistics to be generated. The related sensor may not be available to add to the Energy Dashboard until generation is complete.

Review the [Green Button Component Description](GREEN_BUTTON_COMPONENT_DESCRIPTION.md) for detail on how the Green Button custom component functions.

## Data Imports

Green Button XML blocks do not need to be imported in chronological order, and gaps between imported date ranges are supported. The integration reconciles overlap at the individual interval level: a later import replaces an identical interval, while an ambiguous overlap with a different interval length retains the existing coverage and is logged. Re-importing an identical XML document is ignored.

The import action requires a **Config entry**; select the Green Button integration entry that should receive the data. Provide either a file path or pasted XML content, not both. Setup accepts relative file paths resolved from `/config`; when using the import action, an XML file under `/config` can be used directly and a file outside it must be in an allowlisted directory. For example, after adding `/share` to `allowlist_external_dirs`, use `/share/<green_button_xml_file.xml>`.

The import rejects XML documents with no supported readings or summaries. Invalid or unsupported readings are skipped, and missing cost data is not treated as zero-cost data; cost statistics are generated only when the available source data has complete cost coverage.

## Options

Open the Green Button integration's **Configure** page to change these options:

- **Gas usage allocation**: use daily gas readings, or record usage as a single increment for each billing period.
- **Gas cost allocation**: pro-rate a billing cost across its days, or record it as a single monthly increment.
- **Electricity and gas cost power-of-ten multipliers**: fallback scales used only if the XML omits `powerOfTenMultiplier`. A multiplier declared in the XML always takes precedence.

Long gas billing intervals and summary-only gas data are supported. After changing a fallback cost multiplier, run **Recalculate Green Button Cost Statistics** to update existing cost history; XML-declared multipliers continue to take precedence.

### Power-of-Ten Multiplier for Cost
The Green Button spec says that the [default power-of-ten-multiplier for cost](https://www.greenbuttonalliance.org/costandcurrency) is "-5", but providers may choose a different value. The component uses the `powerOfTenMultiplier` declared in the XML whenever it is present. The electricity and gas multiplier options are fallbacks for sources that omit that value.

If a fallback value is changed after Green Button data has been imported, it applies to new applicable imports. To recalculate previously-imported costs that used the fallback, run the *Recalculate Green Button Cost Statistics* action under Developer Tools -> Actions.

## Services/Actions

The following Green Button actions are available under **Developer Tools → Actions**. Each requires a **Config entry** so it affects only the selected Green Button integration. Actions that change imported data or statistics require an administrator.

- Log Green Button Meter Reading Intervals
- Log Stored Green Button XML Info
- Delete Green Button Statistics
- Import Green Button ESPI XML
- Clear Stored Green Button XML Data
- Recalculate Green Button Cost Statistics

**Clear Stored Green Button XML Data** removes the selected entry's stored XML archive and active import history, but leaves existing recorder and Energy Dashboard statistics untouched. Use **Delete Green Button Statistics** to remove the selected display sensor's imported Green Button statistics; re-import the XML data to rebuild them.

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
