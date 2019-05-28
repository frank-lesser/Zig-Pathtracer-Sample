/// Calls print and then flushes the buffer.
pub fn printf(self: *OutStream, comptime format: []const u8, args: ...) void {
    const State = enum {
        Start,
        OpenBrace,
        CloseBrace,
    };

    comptime var start_index: usize = 0;
    comptime var state = State.Start;
    comptime var next_arg: usize = 0;

    inline for (format) |c, i| {
        switch (state) {
            State.Start => switch (c) {
                '{' => {
                    if (start_index < i) try self.write(format[start_index..i]);
                    state = State.OpenBrace;
                },
                '}' => {
                    if (start_index < i) try self.write(format[start_index..i]);
                    state = State.CloseBrace;
                },
                else => {},
            },
            State.OpenBrace => switch (c) {
                '{' => {
                    state = State.Start;
                    start_index = i;
                },
                '}' => {
                    try self.printValue(args[next_arg]);
                    next_arg += 1;
                    state = State.Start;
                    start_index = i + 1;
                },
                else => @compileError("Unknown format character: " ++ c),
            },
            State.CloseBrace => switch (c) {
                '}' => {
                    state = State.Start;
                    start_index = i;
                },
                else => @compileError("Single '}' encountered in format string"),
            },
        }
    }
    comptime {
        if (args.len != next_arg) {
            @compileError("Unused arguments");
        }
        if (state != State.Start) {
            @compileError("Incomplete format string: " ++ format);
        }
    }
    if (start_index < format.len) {
        try self.write(format[start_index..format.len]);
    }
    try self.flush();
}
const std = @import("std");
const io = std.io;
const os = std.os;

var stderr_file: os.File = undefined;
var stderr_file_out_stream: os.File.OutStream = undefined;
var stderr_stream: ?*io.OutStream(os.File.WriteError) = null;
var stderr_mutex = std.Mutex.init();

pub fn getStderrStream() !*io.OutStream(os.File.WriteError) {
    if (stderr_stream) |st| {
        return st;
    } else {
        stderr_file = try io.getStdErr();
        stderr_file_out_stream = stderr_file.outStream();
        const st = &stderr_file_out_stream.stream;
        stderr_stream = st;
        return st;
    }
}

pub fn _warn(comptime fmt: []const u8, args: ...) void {
    const held = stderr_mutex.acquire();
    defer held.release();
    const stderr = getStderrStream() catch return;
    stderr.print(fmt, args) catch return;
}

//const a_number: i32 = 1234;
//const a_string = "foobar";

//pub fn main() void {
//    _warn("here is a string: '{}' here is a number: {}\n", a_string, a_number);
//}
