import logging
import time
import pyvisa
import yaml
import coloredlogs
import numpy as np
import astropy.units as u

log = logging.getLogger('capture_data')


class SpectrumAnalyserException(Exception):
    pass


class DataOutOfRangeException(Exception):
    pass


class SpectrumAnalyserInterface(object):
    def __init__(self, visa_resource, passive=False):
        self._visa_resource = visa_resource
        self._passive = passive
        self._rm = pyvisa.ResourceManager()
        self.reconnect()

    def reconnect(self):
        self._interface = self._rm.open_resource(
            self._visa_resource)

    def check_error(self):
        msg = self._interface.query(":SYST:ERR:ALL?")
        retval = int(msg.split(",")[0])
        if retval == 0:
            return
        elif retval == -222:
            raise DataOutOfRangeException
        else:
            raise SpectrumAnalyserException(
                ("Error detected from spectrum analyser, "
                 "return value of ':SYST:ERR:ALL?; = {}".format(
                    msg)))

    def send_command(self, command):
        if not self._passive:
            log.debug("Sending SCPI command: {}".format(command))
            self._interface.write(command)

    def send_commands(self, commands):
        for command in commands:
            self.send_command(command)
        self.check_error()

    def get_analysis_bandwidth(self):
        return float(self._interface.query(":TRAC:IQ:BWID?")) * u.Hz

    def get_sampling_rate(self):
        return float(self._interface.query(":TRAC:IQ:SRAT?")) * u.Hz

    def set_centre_frequency(self, frequency):
        self.send_command(":SENS:FREQ:CENT {}".format(str(frequency)))
        self.check_error()

    def get_centre_frequency(self):
        return float(self._interface.query(":SENS:FREQ:CENT?")) * u.Hz

    def get_scaling(self):
        return float(self._interface.query(
            "DISP:WIND:SUBW:TRAC:Y:SCAL:RLEV?")) * u.dB(u.mW)


class Measurement(object):
    def __init__(self, config):
        self._tag = config["userTag"]
        self._scpi_commands = config["spectrumAnalyserScpi"]
        fconfig = config["frequencyRange"]
        units = getattr(u, fconfig["units"])
        self._frequency_start = fconfig["start"] * units
        self._frequency_end = fconfig["end"] * units
        sconfig = config["spectrometerParams"]
        units = getattr(u, sconfig["resolutionUnits"])
        self._resolution = sconfig["resolution"] * units
        units = getattr(u, sconfig["integrationTimeUnits"])
        self._integration_time = sconfig["integrationTime"] * units
        self._output_path = sconfig["outputPath"]

    def get_centre_frequencies(self, bandwidth):
        centre_freqs = []
        frequency = self._frequency_start
        while True:
            frequency = frequency + bandwidth/2
            centre_freqs.append(frequency)
            if frequency + bandwidth/2 > self._frequency_end:
                break
        return centre_freqs


class Spectrometer(object):
    def __init__(self):
        pass

    def configure(self, input_nchans, fft_length):
        pass

    def record(self, naccumulate, frequency, tag, output_path):
        pass

    def abort(self):
        pass

    def deconfigure(self):
        pass


class Executor(object):
    def __init__(self, config, dry_run=False):
        self._config = config
        self._dry_run = dry_run
        self._start_interface()

    def _start_interface(self):
        log.debug("Starting spectrum analyser interface")
        subconfig = self._config["spectrumAnalyser"]
        self._interface = SpectrumAnalyserInterface(
            subconfig["visaResource"],
            passive=self._dry_run)

    def init(self):
        log.info("Initialising spectrum analyser")
        self._interface.send_commands(
            self._config["spectrumAnalyser"]["scpiCommands"])

    def write_header(self, fname, cfreq, bw, total_nchans,
                     integration_time, timestamp, tag):
        with open(fname, "w") as f:
            print("Data: ", file=f)
            print("Center Frequency in Hz: {}".format(cfreq.to(u.Hz).value), file=f)
            print("Bandwidth in Hz: {}".format(bw.to(u.Hz).value), file=f)
            print("Number of Channels: {}".format(total_nchans), file=f)
            print("Frequency Spacing: uniform", file=f)
            print("Integration time in milliseconds: {}".format(integration_time.to(u.ms).value), file=f)
            print("Unique Scan ID: {}".format(timestamp), file=f)
            print("Timestamp: {}".format(timestamp), file=f)
            print("User Friendly Name: {}".format(tag), file=f)
            for param in self._config["headerInformation"]:
                print(param, file=f)

    def run_measurement(self, mconfig):
        measurement = Measurement(mconfig)
        log.info("Running measurement: {}".format(
            measurement._tag))

        log.info("Preparing spectrum analyser")
        self._interface.send_commands(mconfig["spectrumAnalyserScpi"])

        sampling_rate = self._interface.get_sampling_rate()
        log.info("Sampling rate: {}".format(str(sampling_rate)))

        analysis_bandwidth = self._interface.get_analysis_bandwidth()
        log.info("Analysis bandwidth: {}".format(str(analysis_bandwidth)))

        scaling_level = self._interface.get_scaling()
        log.info("Scaling level: {}".format(str(scaling_level)))

        log.info("Output directory: {}".format(measurement._output_path))

        # Calculate the required FFT length and number of accumulated
        # spectra required to satisfy the resolution and integration
        # time.
        fsconfig = self._config["firstStageChanneliser"]
        first_stage_nchans = fsconfig["numChannels"]
        log.info("First stage channeliser Nchans: {}".format(
            first_stage_nchans))
        channel_bandwidth = sampling_rate / first_stage_nchans
        log.info("First stage frequency resolution: {}".format(
            channel_bandwidth))
        fft_length = int((channel_bandwidth /
            measurement._resolution).decompose().value)
        # Round FFT length to next power of 2
        fft_length = 2**((fft_length-1).bit_length())
        log.info("Desired second stage channeliser frequency resolution: {}".format(
            measurement._resolution))
        log.info("Second stage channeliser Nchans: {}".format(
            fft_length))
        actual_resolution = channel_bandwidth / fft_length
        log.info("Actual second stage frequency resolution: {}".format(
            actual_resolution))
        # The number of spectra is rounded up to the next whole
        # number
        naccumulate = int(np.ceil(measurement._integration_time
            * actual_resolution).decompose().value)
        log.info("Second stage number of spectra to accumulate: {}".format(
            naccumulate))
        actual_integration_time = (naccumulate / actual_resolution).decompose()
        log.info("Actual integration time: {}".format(actual_integration_time))
        total_nchans = first_stage_nchans * fft_length
        log.info("Total number of channels: {}".format(
            total_nchans))

        # spectrometer = Spectrometer()
        # spectrometer.capture_start()

        frequencies = measurement.get_centre_frequencies(analysis_bandwidth)
        for frequency in frequencies:
            log.info(("Preparing for {:0.01f} measurement with "
                      "centre frequency {:0.03f}").format(
                      measurement._integration_time, frequency))
            try:
                self._interface.set_centre_frequency(frequency)
            except DataOutOfRangeException:
                log.error("Requested frequency outside of valid range")
                log.warning("Skipping remaining frequencies in current range")
                break
            actual_frequency = self._interface.get_centre_frequency()
            log.info("Actual centre frequency set: {}".format(
                str(actual_frequency)))

            timestamp = int(time.time() * 1000)

            header_fname = "{}/{}_{:0.05}_{}.rfi".format(
                measurement._output_path,
                measurement._tag,
                actual_frequency.to(u.MHz).value,
                timestamp)
            self.write_header(header_fname, actual_frequency, sampling_rate, total_nchans,
                              actual_integration_time, timestamp, measurement._tag)
            log.info("Starting recording system")
            #spectrometer.recording_start()
            #time.sleep(measurement._integration_time.to(u.s).value)
            log.info("Stopping recording system")
            log.info("Measurement complete")

    def run_all_measurements(self):
        for measurement_config in self._config["measurementParameters"]:
            self.run_measurement(measurement_config)


def parse_config(config_file):
    log.info("Parsing configuration from file: {}".format(config_file))
    with open(config_file, "r") as f:
        try:
            config = yaml.load(f)
        except Exception as error:
            log.exception("Error during configuration file load")
            raise error
        else:
            log.debug("Parsed config: {}".format(config))
            return config


def main(config_file, dry_run):
    config = parse_config(config_file)
    executor = Executor(config, dry_run=dry_run)
    executor.init()
    if 'measurementParameters' in config:
        executor.run_all_measurements()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Perform a data capture from the RFI chamber recording system')
    parser.add_argument('--config', metavar='FILE', type=str,
        required=True, help='The YAML measurement configuration file')
    parser.add_argument('--dry-run', action="store_true",
        help='Do not send configuration requests to the spetrum analyser only queries')
    parser.add_argument('--log-level', metavar='LEVEL', type=str,
        default="INFO", help='The logging level ({})'.format(
            ", ".join(logging.getLevelName(ii) for ii in range(10, 60, 10))))
    args = parser.parse_args()
    coloredlogs.install(
        fmt="[ %(levelname)s - %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
        level=args.log_level.upper(),
        logger=log)
    main(args.config, args.dry_run)
