#!/usr/bin/env python3

import logging
import time
import pyvisa
import yaml
import coloredlogs
import sys
import os
import json
import numpy as np
from subprocess import Popen, PIPE
from threading import Thread, Event
import astropy.units as u

log = logging.getLogger('capture_data')
MAX_FFT_LENGTH = 1<<28
DADA_BLOCK_SIZE = 1073741824#8589934592
DADA_NBLOCKS = 12
DADA_KEY = "dada"
MKRECV_FILE_PATH = "/tmp/mkrecv.cfg"
MKRECV_CONF_PFB_MODE = """
HEADER       DADA                # Distributed aquisition and data analysis
HDR_VERSION  1.0                 # Version of this ASCII header
HDR_SIZE     4096                # Size of the header in bytes
DADA_VERSION 1.0                 # Version of the DADA Software

# time of the rising edge of the first time sample
UTC_START    unset               # yyyy-mm-dd-hh:mm:ss.fs
MJD_START    unset               # MJD equivalent to the start UTC

#MeerKAT specifics
DADA_KEY     dada
SYNC_TIME    1231235243.0000000
SAMPLE_CLOCK 1750000000.0
MCAST_SOURCES 225.0.0.100+15 #,225.0.0.153,225.0.0.154,225.0.0.155
PORT         7148
#UDP_IF       10.10.1.11
IBV_IF      192.168.2.81
IBV_VECTOR   -1
IBV_MAX_POLL 10
#SAMPLE_CLOCK_START 0
HEAP_NBYTES 8192
PACKET_SIZE 9000
BUFFER_SIZE 128000000
DADA_NSLOTS 4
NTHREADS 9

#MeerKat F-Engine
NINDICES    2
# The first index item is the running timestamp
IDX1_ITEM   0         # First item of a SPEAD heap
IDX1_STEP   1   # The difference between successive timestamps
# The second index item distinguish between both polarizations
IDX2_ITEM   2
IDX2_LIST   0:16
"""

MKRECV_CONF_PASSTHROUGH_MODE = """
HEADER       DADA                # Distributed aquisition and data analysis
HDR_VERSION  1.0                 # Version of this ASCII header
HDR_SIZE     4096                # Size of the header in bytes
DADA_VERSION 1.0                 # Version of the DADA Software

# time of the rising edge of the first time sample
UTC_START    unset               # yyyy-mm-dd-hh:mm:ss.fs
MJD_START    unset               # MJD equivalent to the start UTC

#MeerKAT specifics
DADA_KEY     dada
SYNC_TIME    1231235243.0000000
SAMPLE_CLOCK 1750000000.0
MCAST_SOURCES 225.0.0.100+15 #,225.0.0.153,225.0.0.154,225.0.0.155
PORT         7148
#UDP_IF       10.10.1.11
IBV_IF       192.168.2.81 
IBV_VECTOR   -1
IBV_MAX_POLL 10
#SAMPLE_CLOCK_START 0
HEAP_NBYTES 8192
PACKET_SIZE 9000
BUFFER_SIZE 128000000
DADA_NSLOTS 4
NTHREADS 9

#MeerKat F-Engine
NINDICES    1
# The first index item is the running timestamp
IDX1_ITEM   0         # First item of a SPEAD heap
IDX1_STEP   1   # The difference between successive timestamps
"""


class SpectrumAnalyserException(Exception):
    pass


class DataOutOfRangeException(Exception):
    pass


class PipeHandler(Thread):
    def __init__(self, pipe):
        Thread.__init__(self)
        self._pipe = pipe
        self._stop_event = Event()
        self.setDaemon(True)
        self.start()

    def stop(self):
        self._stop_event.set()
        self.join()


class MKRECVStdoutHandler(PipeHandler):
    def __init__(self, pipe, nskip):
        self._nskip = nskip
        PipeHandler.__init__(self, pipe)

    def parse_stat_line(self, line):
        sp = line.split()
        total_slots = int(sp[1])
        filled_slots = int(sp[3])
        if total_slots != filled_slots:
            lost_fraction = 1 - float(filled_slots) / total_slots
            log.warning(("Packet loss detected in network capture ({:0.06f}% loss) "
                         "consider repeating this measurement").format(
                         100.0 * lost_fraction))

    def run(self):
        while not self._stop_event.is_set():
            line = self._pipe.readline()
            log.debug("{}".format(line))
            if line.startswith(b"STAT"):
                self._nskip -= 1
                if self._nskip > 0:
                    continue
                else:
                    self.parse_stat_line(line)


class RSSpectrometerStdoutHandler(PipeHandler):
    def __init__(self, pipe):
        PipeHandler.__init__(self, pipe)

    def run(self):
        while not self._stop_event.is_set():
            line = self._pipe.readline()
            log.debug("{}".format(line))
            if b"[info]" in line:
                log.info(line.decode().strip("\n"))
            elif b"[error]" in line:
                log.error(line.decode().strip("\n"))


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
        frequency = frequency + bandwidth/2
        centre_freqs.append(frequency)
        while True:
            if frequency + bandwidth/2 > self._frequency_end:
                break
            frequency = frequency + bandwidth
            centre_freqs.append(frequency)
        return centre_freqs


def syscmd_wrapper(cmd):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    if proc.wait() != 0:
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        raise Exception("Command: '{}' failed\nstdout: {}\nstderr: {}".format(
            " ".join(cmd), stdout, stderr))


class Spectrometer(object):
    def __init__(self):
        self._mkrecv_proc = None
        self._spec_proc = None
        self._nskip = 4

    def configure(self):
        """
        # Destroy any previous DADA buffers
        log.debug("Cleaning up any previous DADA buffers")
        try:
            syscmd_wrapper(["taskset", "-c", "10-19", "dada_db", "-k", DADA_KEY, "-d"])
        except Exception as e:
            pass

        # Create new DADA buffer
        log.debug("Allocating DADA buffer")
        syscmd_wrapper(["taskset", "-c", "10-19", "dada_db",
                        "-k", DADA_KEY,
                        "-b", str(DADA_BLOCK_SIZE),
                        "-n", str(DADA_NBLOCKS),
                        "-l", "-p"])
        """
        

    def record(self, input_nchans, fft_length, naccumulate, output_file, reference_level):
        log.debug("Writing MKRECV header file")
        with open(MKRECV_FILE_PATH, "w") as f:
            if input_nchans == 1:
                log.info("Assuming PASSTHROUGH mode on FPGA")
                f.write(MKRECV_CONF_PASSTHROUGH_MODE)
            else:
                log.info("Assuming PFB mode on FPGA")
                f.write(MKRECV_CONF_PFB_MODE)
        #log.debug("Reseting DADA buffer")
        #syscmd_wrapper(["taskset", "-c", "10-19", "dbreset", "-k", DADA_KEY])
        
        
        # Destroy any previous DADA buffers
        log.debug("Cleaning up any previous DADA buffers")
        try:
            syscmd_wrapper(["taskset", "-c", "0-9", "dada_db", "-k", DADA_KEY, "-d"])
        except Exception as e:
            pass

        # Create new DADA buffer
        log.debug("Allocating DADA buffer")
        syscmd_wrapper(["taskset", "-c", "0-9", "dada_db",
                        "-k", DADA_KEY,
                        "-b", str(DADA_BLOCK_SIZE),
                        "-n", str(DADA_NBLOCKS),
                        "-l", "-p"])
        
        log.debug("Starting spectrometer")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        self._spec_proc = Popen([
            "taskset", "-c", "9",
            "rsspectrometer",
            "--key", DADA_KEY,
            "--input-nchans", str(input_nchans),
            "--fft-length", str(fft_length),
            "--naccumulate", str(naccumulate),
            "--reflevel", str(reference_level.value),
            "--nskip", str(self._nskip),
            "-o", output_file,
            "--log-level", "info"],
            stdout=sys.stdout, stderr=sys.stderr, bufsize=1)
        
        #self._spec_proc = Popen(["dbnull"])
        log.debug("Starting mkrecv")
        self._mkrecv_proc = Popen([
            "taskset", "-c", "0-8",
            "mkrecv_rnt", "--header", MKRECV_FILE_PATH,
            "--slots-skip","4","--quiet"],
            stdout=PIPE, stderr=sys.stderr, bufsize=1)
        mkrecv_monitor = MKRECVStdoutHandler(self._mkrecv_proc.stdout, self._nskip)
        #rs_monitor = RSSpectrometerStdoutHandler(self._spec_proc.stdout)
        self._spec_proc.wait()
        self._mkrecv_proc.terminate()
        mkrecv_monitor.stop()
        #rs_monitor.stop()


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
        # read analysis band:
        abw = self._interface.get_analysis_bandwidth()
        header_dict = {
            "Center Frequency in Hz": cfreq.to(u.Hz).value,
            "Analysis Center Frequency in Hz":cfreq.to(u.Hz).value,
            "Bandwidth in Hz": bw.to(u.Hz).value,
            "Analysis Bandwidth in Hz":abw.to(u.Hz).value,
            "Center Frequency in Hz":cfreq.to(u.Hz).value,
            "Bandwidth in Hz":bw.to(u.Hz).value,
            "Number of Channels": total_nchans,
            "Frequency Spacing": "uniform",
            "Integration time in milliseconds": integration_time.to(u.ms).value,
            "Unique Scan ID": fname.split("/")[-1].strip(".rfi"),
            "Timestamp": timestamp,
            "User Friendly Name": tag

        }
        for param in self._config["headerInformation"]:
            header_dict[param["key"]] = param["value"]
        with open(fname, "w") as f:
            json.dump(header_dict, f)

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
        output_dir = "/".join((measurement._output_path, time.strftime("%Y%m%d-%H%M%S/")))
        log.info("Output directory: {}".format(output_dir))

        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass
        except Exception as error:
            log.exception("Cannot create output directory")
            raise error
    
        try:
            os.chown(output_dir, 1000, 1000)
        except Exception as error:
            log.exception("Cannot CHOWN output directory to rfiops")
            raise error



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
        if fft_length > MAX_FFT_LENGTH:
            message = "Resolution exceeds maximum FFT length ({} pts)".format(MAX_FFT_LENGTH)
            log.error(message)
            raise Exception(message)
        
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

        spectrometer = Spectrometer()
        spectrometer.configure()
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
            filename_stem = "{}/{}_{:0.05}_{}".format(
                output_dir,
                measurement._tag,
                actual_frequency.to(u.MHz).value,
                timestamp)
            data_fname = "{}.npy".format(filename_stem)
            header_fname = "{}.rfi".format(filename_stem)
            self.write_header(header_fname, actual_frequency, sampling_rate, total_nchans,
                              actual_integration_time, timestamp, measurement._tag)
            log.info("Starting recording system")
            spectrometer.record(first_stage_nchans, fft_length, naccumulate, data_fname, scaling_level)
            log.info("Recording done")
            log.info("Measurement complete")

    def run_all_measurements(self):
        for measurement_config in self._config["measurementParameters"]:
            try:
                self.run_measurement(measurement_config)
            except Exception as error:
                log.error("Measurement failed with error '{}', skipping to next measurement".format(
                    str(error)))


def parse_config(config_file):
    log.info("Parsing configuration from file: {}".format(config_file))
    with open(config_file, "r") as f:
        try:
            config = yaml.full_load(f)
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
    parser.add_argument('--log-dir', metavar='DIR', type=str,
        help='A directory to output logs to, if no directory is specified logs will only go to stdout')
    args = parser.parse_args()
    coloredlogs.install(
        fmt="[ %(levelname)s - %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
        level=args.log_level.upper(),
        logger=log)

    if args.log_dir is not None:
        try:
            os.makedirs(args.log_dir)
        except FileExistsError:
            pass
        except Exception:
            log.exception("Error while creating logging directory")
        log_file = "{}/{}".format(args.log_dir, time.strftime("%Y-%m-%dT%H:%M:%S_rfi_chamber.log"))
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter("[ %(levelname)s - %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s")
        fh.setFormatter(formatter)
        log.addHandler(fh)
        log.info("Log file: {}".format(log_file))

    try:
        main(args.config, args.dry_run)
    except KeyboardInterrupt:
        log.warning("User Ctrl-C interrupt")
    except Exception as error:
        log.error("Exception '{}' propagated to top of stack, cleaning up DADA buffers and exiting.".format(
            str(error)))
    finally:
        log.info("Cleaning up shared memory")
        try:
            syscmd_wrapper(["dada_db", "-k", DADA_KEY, "-d"])
        except:
            pass
        log.info("Cleaning up any hanging capture instances")
        try:
            syscmd_wrapper(["pkill", "--signal", "9", "mkrecv_rnt"])
            syscmd_wrapper(["pkill", "--signal", "9", "rsspectrometer"])
        except:
            pass
