/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::Ref.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_REF_HPP
#define OPENKALMAN_EIGEN3_TRAITS_REF_HPP

#include <type_traits>


namespace OpenKalman::interface
{
#ifndef __cpp_concepts
  template<typename PlainObjectType, int Options, typename StrideType>
  struct IndexTraits<Eigen::Ref<PlainObjectType, Options, StrideType>>
    : detail::IndexTraits_Eigen_default<Eigen::Ref<PlainObjectType, Options, StrideType>> {};
#endif


  template<typename PlainObjectType, int Options, typename StrideType>
  struct Dependencies<Eigen::Ref<PlainObjectType, Options, StrideType>>
  {
    static constexpr bool has_runtime_parameters = false;
    // Ref is not self-contained in any circumstances.
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_REF_HPP
