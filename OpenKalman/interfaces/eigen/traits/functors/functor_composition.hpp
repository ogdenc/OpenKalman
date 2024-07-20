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
 * \brief Definition for Eigen3::functor_composition.
 */

#ifndef OPENKALMAN_EIGEN_FUNCTOR_COMPOSITION_HPP
#define OPENKALMAN_EIGEN_FUNCTOR_COMPOSITION_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  /**
   * Compose two Eigen functors F1 and F2. (F2 is applied first).
   * \tparam F1
   * \tparam F2
   */
  template<typename F1, typename F2>
  struct functor_composition
  {
  private:

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct maybe_constexpr { using type = T; };


#ifdef __cpp_concepts
    template<constexpr_unary_operation_defined T>
    struct maybe_constexpr<T>
#else
    template<typename T>
    struct maybe_constexpr<T, std::enable_if_t<constexpr_unary_operation_defined<T>>>
#endif
    {
      using type = decltype(UnaryFunctorTraits<std::decay_t<T>>::constexpr_operation());
    };


    using CF1 = typename maybe_constexpr<F1>::type;
    using CF2 = typename maybe_constexpr<F2>::type;

  public:

#ifdef __cpp_concepts
    constexpr functor_composition() requires std::default_initializable<CF1> and std::default_initializable<CF2> = default;
#else
    template<typename X = CF1, std::enable_if_t<std::is_default_constructible_v<X> and std::is_default_constructible_v<CF2>, int> = 0>
    constexpr functor_composition() {};
#endif

  private:

    template<typename Arg>
    static constexpr decltype(auto)
    get_constexpr_op(Arg&& arg)
    {
      if constexpr (constexpr_unary_operation_defined<Arg>) return UnaryFunctorTraits<std::decay_t<Arg>>::constexpr_operation();
      else return std::forward<Arg>(arg);
    }

  public:

    template<typename MF1, typename MF2>
    constexpr functor_composition(MF1&& f1, MF2&& f2)
      : m_functor1 {get_constexpr_op(std::forward<MF1>(f1))}, m_functor2{get_constexpr_op(std::forward<MF2>(f2))} {}


    template<typename Scalar>
    constexpr auto operator()(const Scalar& a) const { return m_functor1(m_functor2(a)); }


    template<typename Packet>
    constexpr auto packetOp(const Packet& a) const { return m_functor1.packetOp(m_functor2.packetOp(a)); }


    constexpr decltype(auto) functor1() const & { return m_functor1; }
    constexpr decltype(auto) functor1() const && { return std::forward<CF1>(m_functor1); }

    constexpr decltype(auto) functor2() const & { return m_functor2; }
    constexpr decltype(auto) functor2() const && { return std::forward<CF2>(m_functor2); }

  private:

    CF1 m_functor1;
    CF2 m_functor2;

  };


  /**
   * Deduction guide for \ref functor_composition.
   */
  template<typename F1, typename F2>
  functor_composition(F1&&, F2&&) -> functor_composition<F1, F2>;


  // UnaryFunctorTraits for functor_composition
  template<typename F1, typename F2>
  struct UnaryFunctorTraits<functor_composition<F1, F2>>
  {
  private:

    using UnaryFunctorTraits1 = UnaryFunctorTraits<std::decay_t<F1>>;
    using UnaryFunctorTraits2 = UnaryFunctorTraits<std::decay_t<F2>>;

  public:

    static constexpr bool preserves_triangle = UnaryFunctorTraits1::preserves_triangle and UnaryFunctorTraits2::preserves_triangle;
    static constexpr bool preserves_hermitian = UnaryFunctorTraits1::preserves_hermitian and UnaryFunctorTraits2::preserves_hermitian;

    template<typename UnaryOp, typename XprType>
    static constexpr auto get_constant(const Eigen::CwiseUnaryOp<UnaryOp, XprType>& arg)
    {
      if constexpr (constant_matrix<XprType, ConstantType::static_constant>)
        return values::scalar_constant_operation<UnaryOp, constant_coefficient<XprType>>{};
      else
        return values::scalar_constant_operation {arg.functor(), constant_coefficient {arg.nestedExpression()}};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return values::scalar_constant_operation {arg.functor(), constant_diagonal_coefficient {arg.nestedExpression()}};
    }

  };

} // namespace OpenKalman::Eigen3


namespace Eigen::internal
{
  template<typename F1, typename F2>
  struct functor_traits<OpenKalman::Eigen3::functor_composition<F1, F2>>
  {
    enum
    {
      Cost = int{functor_traits<std::decay_t<F1>>::Cost} + int{functor_traits<std::decay_t<F2>>::Cost},
      PacketAccess = functor_traits<std::decay_t<F1>>::PacketAccess and functor_traits<std::decay_t<F2>>::PacketAccess,
    };
  };

} // namespace Eigen::internal


#endif //OPENKALMAN_EIGEN_FUNCTOR_COMPOSITION_HPP
