/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::TensorMap.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORMAP_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORMAP_HPP


namespace OpenKalman::interface
{
  template<typename PlainObjectType, int Options, template<typename> typename MakePointer>
  struct object_traits<Eigen::TensorMap<PlainObjectType, Options, MakePointer>>
    : Eigen3::object_traits_tensor_base<Eigen::TensorMap<PlainObjectType, Options, MakePointer>>
  {
  private:

    using Xpr = Eigen::TensorMap<PlainObjectType, Options, MakePointer>;
    using Base = Eigen3::object_traits_tensor_base<Xpr>;
    using StorageRefType = typename Xpr::StorageRefType;
    using IndexType = typename Xpr::Index;

  public:

    template<typename Arg, typename N>
    static constexpr std::size_t get_pattern_collection(const Arg& arg, N n) { return arg.dimension(n); }

    // nested_object() not defined

    // get_constant() not defined

    // get_constant_diagonal() not defined


#ifdef __cpp_lib_concepts
    template<typename Arg, std::convertible_to<IndexType>...I> requires (sizeof...(I) == PlainObjectType::NumDimensions)
#else
    template<typename Arg, typename...I, std::enable_if_t<(stdex::convertible_to<I, IndexType> and ...) and
      (sizeof...(I) == PlainObjectType::NumDimensions), int> = 0>
#endif
    static constexpr decltype(auto) get(Arg&& arg, I...i)
    {
      return std::forward<Arg>(arg)(static_cast<IndexType>(i)...);
    }


#ifdef __cpp_lib_concepts
    template<typename Arg, std::convertible_to<IndexType>...I> requires (sizeof...(I) == PlainObjectType::NumDimensions) and
    std::is_lvalue_reference_v<StorageRefType> and (not std::is_const_v<std::remove_reference_t<StorageRefType>>)
#else
    template<typename Arg, typename...I, std::enable_if_t<(stdex::convertible_to<I, IndexType> and ...) and
      (sizeof...(I) == PlainObjectType::NumDimensions) and std::is_lvalue_reference<StorageRefType>::value and
      not std::is_const<typename std::remove_reference<StorageRefType>::type>::value, int> = 0>
#endif
    static void set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
    {
      arg(static_cast<IndexType>(i)...) = s;
    }

    static constexpr bool is_writable = false;

    template<typename Arg>
    static constexpr auto * const
    raw_data(Arg& arg) { return arg.data(); }

    static constexpr data_layout layout = layout_of_v<PlainObjectType>;

  };

}

#endif
