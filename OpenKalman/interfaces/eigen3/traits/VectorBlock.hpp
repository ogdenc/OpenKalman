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
 * \brief Type traits as applied to Eigen::VectorBlock.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_VECTORBLOCK_HPP
#define OPENKALMAN_EIGEN3_TRAITS_VECTORBLOCK_HPP

#include <type_traits>


namespace OpenKalman::interface
{
#ifndef __cpp_concepts
  template<typename VectorType, int Size>
  struct IndexTraits<Eigen::VectorBlock<VectorType, Size>>
    : detail::IndexTraits_Eigen_default<Eigen::Ref<Eigen::VectorBlock<VectorType, Size>>> {};
#endif


  template<typename VectorType, int Size>
  struct Dependencies<Eigen::VectorBlock<VectorType, Size>>
  {
    static constexpr bool has_runtime_parameters = true;
    using type = std::tuple<typename Eigen::internal::ref_selector<VectorType>::non_const_type>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    // Eigen::VectorBlock should always be converted to Matrix

  };


  template<typename VectorType, int Size>
  struct SingleConstant<Eigen::VectorBlock<VectorType, Size>>
  {
    const Eigen::VectorBlock<VectorType, Size>& xpr;

    constexpr auto get_constant()
    {
      return constant_coefficient {xpr.nestedExpression()};
    }
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_VECTORBLOCK_HPP
