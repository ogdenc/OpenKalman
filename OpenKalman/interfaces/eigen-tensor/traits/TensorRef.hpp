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
 * \brief Type traits as applied to Eigen::TensorRef.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORREF_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORREF_HPP


namespace OpenKalman::interface
{
  template<typename PlainObjectType>
  struct IndexibleObjectTraits<Eigen::TensorRef<PlainObjectType>>
    : Eigen3::indexible_object_traits_base<Eigen::TensorRef<PlainObjectType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::TensorRef<PlainObjectType>>;
    using Base::max_indices;
    using Dimensions = PlainObjectType::Dimensions;
    using Scalar = Eigen::internal::traits<PlainObjectType>::Scalar;
    using Index = Eigen::internal::traits<PlainObjectType>::Index;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_index_descriptor(const Arg& arg, N n) { return arg.dimension(n); }

    static constexpr bool has_runtime_parameters = false;

    using dependents = std::tuple<Eigen::internal::TensorLazyBaseEvaluator<Dimensions, Scalar>*>;

    // get_nested_matrix() not defined

    // convert_to_self_contained() not defined

    // get_constant() not defined

    // get_constant_diagonal() not defined


#ifdef __cpp_lib_concepts
    template<typename Arg, std::convertible_to<Index>...I> requires (sizeof...(I) == max_indices_of_v<PlainObjectType>)
#else
    template<typename Arg, typename...I, std::enable_if_t<(std::is_convertible_v<I, Index> and ...) and
      (sizeof...(I) == max_indices_of_v<PlainObjectType>), int> = 0>
#endif
    static constexpr decltype(auto) get(Arg&& arg, I...i)
    {
      if constexpr ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0)
        return std::forward<Arg>(arg).coeffRef(std::array<Index, sizeof...(I)>{static_cast<Index>(i)...});
      else
        return std::forward<Arg>(arg).coeff(std::array<Index, sizeof...(I)>{static_cast<Index>(i)...});
    }


#ifdef __cpp_lib_concepts
    template<typename Arg, std::convertible_to<Index>...I> requires (sizeof...(I) == max_indices_of_v<PlainObjectType>) and
      ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0x0)
#else
    template<typename Arg, typename...I, std::enable_if_t<(std::is_convertible_v<I, Index> and ...) and
      (sizeof...(I) == max_indices_of_v<PlainObjectType>) and ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0x0), int> = 0>
#endif
    static void set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
    {
      arg.coeffRef(std::array<Index, sizeof...(I)>{static_cast<Index>(i)...}) = s;
    }

    static constexpr bool is_writable = true;

    // data() not defined

    static constexpr Layout layout = layout_of_v<PlainObjectType>;

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORREF_HPP
