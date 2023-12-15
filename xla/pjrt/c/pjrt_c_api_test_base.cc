/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/pjrt/c/pjrt_c_api_test_base.h"

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace pjrt {
namespace {

PJRT_Client* CreateClient(const PJRT_Api* api) {
  PJRT_Client_Create_Args create_args;
  create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_args.priv = nullptr;
  create_args.create_options = nullptr;
  create_args.num_options = 0;
  create_args.kv_get_callback = nullptr;
  create_args.kv_put_callback = nullptr;
  create_args.kv_put_user_arg = nullptr;
  create_args.kv_get_user_arg = nullptr;
  PJRT_Error* error = api->PJRT_Client_Create(&create_args);
  CHECK_EQ(error, nullptr);
  CHECK_NE(create_args.client, nullptr);
  return create_args.client;
}

}  // namespace

PjrtCApiTestBase::PjrtCApiTestBase(const PJRT_Api* api) {
  api_ = api;
  client_ = CreateClient(api_);
  executable_ = Create_Executable(api_, client_, CreateAddOneComputation());
}

PjrtCApiTestBase::~PjrtCApiTestBase() {
  executable_.reset();
  destroy_client(client_);
}

void PjrtCApiTestBase::destroy_client(PJRT_Client* client) {
  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.priv = nullptr;
  destroy_args.client = client;
  PJRT_Error* error = api_->PJRT_Client_Destroy(&destroy_args);
  CHECK_EQ(error, nullptr);
}

int PjrtCApiTestBase::GetDeviceId(PJRT_DeviceDescription* device_desc) const {
  PJRT_DeviceDescription_Id_Args args = PJRT_DeviceDescription_Id_Args{
      .struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE,
      .priv = nullptr,
      .device_description = device_desc,
      .id = -1,
  };
  PJRT_Error* error = api_->PJRT_DeviceDescription_Id(&args);
  CHECK_EQ(error, nullptr);
  return args.id;
}

int PjrtCApiTestBase::GetDeviceId(PJRT_Device* device) const {
  return GetDeviceId(::pjrt::GetDeviceDescription(api_, device));
}

bool PjrtCApiTestBase::IsValidDeviceId(PJRT_Device* device) const {
  return GetDeviceId(device) >= 0;
}

int PjrtCApiTestBase::GetLocalHardwareId(PJRT_Device* device) const {
  PJRT_Device_LocalHardwareId_Args args = PJRT_Device_LocalHardwareId_Args{
      .struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE,
      .priv = nullptr,
      .device = device,
      .local_hardware_id = -1,
  };
  PJRT_Error* error = api_->PJRT_Device_LocalHardwareId(&args);
  CHECK_EQ(error, nullptr);
  return args.local_hardware_id;
}

absl::Span<PJRT_Device* const> PjrtCApiTestBase::GetClientDevices() const {
  PJRT_Client_Devices_Args dev_args;
  dev_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  dev_args.priv = nullptr;
  dev_args.client = client_;
  PJRT_Error* error = api_->PJRT_Client_Devices(&dev_args);
  CHECK(error == nullptr);
  return absl::MakeSpan(dev_args.devices, dev_args.num_devices);
}

int PjrtCApiTestBase::GetNumDevices() const {
  return GetClientDevices().size();
}

std::string PjrtCApiTestBase::BuildSingleDeviceCompileOptionStr() {
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(0);
  xla::DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = 0;
  build_options.set_device_assignment(device_assignment);
  xla::CompileOptions options;
  options.executable_build_options = build_options;
  absl::StatusOr<xla::CompileOptionsProto> options_proto = options.ToProto();
  TF_CHECK_OK(options_proto.status());
  return options_proto->SerializeAsString();
}

absl::Span<PJRT_Device* const> PjrtCApiTestBase::GetClientAddressableDevices()
    const {
  PJRT_Client_AddressableDevices_Args addr_args;
  addr_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  addr_args.priv = nullptr;
  addr_args.client = client_;
  PJRT_Error* error = api_->PJRT_Client_AddressableDevices(&addr_args);
  CHECK(error == nullptr);
  return absl::MakeSpan(addr_args.addressable_devices,
                        addr_args.num_addressable_devices);
}

PJRT_Client_BufferFromHostBuffer_Args
PjrtCApiTestBase::CreateBufferFromHostBufferArgs(
    const std::vector<float>& data, const xla::Shape& shape,
    const xla::PjRtClient::HostBufferSemantics host_buffer_semantics,
    PJRT_Device* device) {
  PJRT_Client_BufferFromHostBuffer_Args args;
  args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  args.priv = nullptr;

  args.data = data.data();
  args.type = ::pjrt::ConvertToPjRtBufferType(shape.element_type());
  args.dims = shape.dimensions().data();
  args.num_dims = shape.dimensions().size();
  args.byte_strides = nullptr;
  args.num_byte_strides = 0;
  args.device_layout = nullptr;
  args.host_buffer_semantics =
      ::pjrt::ConvertToPjRtHostBufferSemantics(host_buffer_semantics);
  args.client = client_;
  if (device == nullptr) {
    device = GetClientAddressableDevices()[0];
  }
  args.device = device;
  args.memory = nullptr;
  return args;
}

std::pair<std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter>,
          xla::PjRtFuture<absl::Status>>
PjrtCApiTestBase::create_buffer(PJRT_Device* device) {
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithType<float>({4});
  std::vector<float> float_data(4);
  std::iota(float_data.begin(), float_data.end(), 41.0f);

  PJRT_Client_BufferFromHostBuffer_Args args = CreateBufferFromHostBufferArgs(
      float_data, shape,
      xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, device);

  auto transfer_error =
      ToUniquePtr(api_->PJRT_Client_BufferFromHostBuffer(&args));
  EXPECT_EQ(transfer_error, nullptr);

  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
      args.buffer, ::pjrt::MakeBufferDeleter(api_));

  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter>
      done_with_host_buffer_event(args.done_with_host_buffer,
                                  ::pjrt::MakeEventDeleter(api_));

  PJRT_Buffer_ReadyEvent_Args get_event_args;
  get_event_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
  get_event_args.priv = nullptr;
  get_event_args.buffer = buffer.get();
  auto ready_event_error =
      ToUniquePtr(api_->PJRT_Buffer_ReadyEvent(&get_event_args));
  EXPECT_EQ(ready_event_error, nullptr);
  xla::PjRtFuture<absl::Status> buffer_ready_event =
      ::pjrt::ConvertCEventToCppFuture(get_event_args.event, api_);

  return std::make_pair(std::move(buffer), buffer_ready_event);
}

std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter>
PjrtCApiTestBase::ToUniquePtr(PJRT_Error* error) {
  return std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter>{
      error, ::pjrt::MakeErrorDeleter(api_)};
}

xla::XlaComputation PjrtCApiTestBase::CreateAddOneComputation() {
  xla::XlaBuilder builder(std::string{kExecutableName});
  xla::Shape s = xla::ShapeUtil::MakeShape(xla::F32, {});
  auto inp = Parameter(&builder, 0, s, "input");
  auto one = xla::ConstantR0<float>(&builder, 1.0f);
  auto incremented = Add(inp, one);
  return builder.Build(incremented).value();
}

std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
PjrtCApiTestBase::Create_Executable(const PJRT_Api* c_api, PJRT_Client* client,
                                    const xla::XlaComputation& computation) {
  std::string module_str = computation.proto().SerializeAsString();
  std::string format(::pjrt::kHloFormat);
  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.priv = nullptr;
  program.code = module_str.data();
  program.code_size = module_str.size();
  program.format = format.data();
  program.format_size = format.size();

  xla::CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(1);
  absl::StatusOr<xla::CompileOptionsProto> compile_options_proto =
      compile_options.ToProto();
  TF_CHECK_OK(compile_options_proto.status());

  PJRT_Client_Compile_Args compile_args;
  compile_args.struct_size = sizeof(PJRT_Client_Compile_Args);
  compile_args.priv = nullptr;
  compile_args.program = std::move(&program);
  compile_args.client = client;
  compile_args.compile_options =
      compile_options_proto->SerializeAsString().c_str();
  compile_args.compile_options_size = compile_options_proto->ByteSizeLong();
  compile_args.executable = nullptr;

  PJRT_Error* client_compile_error = c_api->PJRT_Client_Compile(&compile_args);
  CHECK_EQ(client_compile_error, nullptr);

  std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
      unique_executable(compile_args.executable,
                        ::pjrt::MakeLoadedExecutableDeleter(c_api));

  return unique_executable;
}

std::unique_ptr<PJRT_Executable, ::pjrt::PJRT_ExecutableDeleter>
PjrtCApiTestBase::GetExecutable(PJRT_LoadedExecutable* loaded_executable,
                                const PJRT_Api* api) {
  PJRT_LoadedExecutable_GetExecutable_Args args;
  args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.loaded_executable = loaded_executable;
  args.executable = nullptr;
  ::pjrt::LogFatalIfPjrtError(api->PJRT_LoadedExecutable_GetExecutable(&args),
                              api);
  return {args.executable, ::pjrt::MakeExecutableDeleter(api)};
}

}  // namespace pjrt
