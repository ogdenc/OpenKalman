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
 * \internal
 * \file
 * \brief Definitions for Eigen3::EigenWrapper
 */

#ifndef OPENKALMAN_EIGENWRAPPER_HPP
#define OPENKALMAN_EIGENWRAPPER_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Deduction guide for EigenWrapper
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_general Arg> requires (not Eigen3::eigen_wrapper<Arg>)
#else
  template<typename Arg, std::enable_if_t<Eigen3::eigen_general<Arg> and not Eigen3::eigen_wrapper<Arg>, int> = 0>
#endif
  LibraryWrapper(Arg&&) -> LibraryWrapper<Arg, Eigen::Matrix<scalar_type_of_t<Arg>, 0, 0>>;

} // namespace OpenKalman


namespace Eigen::internal
{
  // ------------------------- //
  //  Eigen::internal::traits  //
  // ------------------------- //

  namespace OpenKalman_detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct EigenWrapperTraits
    {
    private:

      using ElementRef = decltype(OpenKalman::get_element(std::declval<T&>(), 0, 0));
      static constexpr auto layout_bit = OpenKalman::layout_of_v<T> == OpenKalman::Layout::right ? RowMajorBit : 0x0;
      static constexpr auto direct = OpenKalman::directly_accessible<T> ? DirectAccessBit : 0x0;
      static constexpr bool lvalue_get_element = std::is_same_v<ElementRef, std::decay_t<ElementRef>&>;

#ifdef __cpp_concepts
      template<typename Arg>
#else
      template<typename Arg, typename = void>
#endif
      struct Strides { using type = std::decay_t<decltype(OpenKalman::internal::strides(std::declval<Arg>()))>; };

#ifdef __cpp_concepts
      template<typename Arg> requires (OpenKalman::layout_of_v<Arg> == OpenKalman::Layout::none)
      struct Strides<Arg>
#else
      template<typename Arg>
      struct Strides<Arg, std::enable_if_t<(OpenKalman::layout_of_v<Arg> == OpenKalman::Layout::none)>>
#endif
      { using type = std::tuple<std::size_t, std::size_t>; };

      using Stride0 = std::tuple_element_t<0, typename Strides<T>::type>;
      using Stride1 = std::tuple_element_t<1, typename Strides<T>::type>;

      using IndexType = typename T::Index;

      static constexpr IndexType stride0 = OpenKalman::static_index_value<Stride0> ? OpenKalman::static_index_value_of_v<Stride0> : Eigen::Dynamic;
      static constexpr IndexType stride1 = OpenKalman::static_index_value<Stride1> ? OpenKalman::static_index_value_of_v<Stride1> : Eigen::Dynamic;

    public:

      using XprKind = MatrixXpr;
      enum {
        Flags = layout_bit | direct | (lvalue_get_element ? LvalueBit : 0x0) | (std::is_lvalue_reference_v<T> ? 0x0 : NestByRefBit),
        InnerStrideAtCompileTime = layout_bit == RowMajorBit ? stride1 : stride0,
        OuterStrideAtCompileTime = layout_bit == RowMajorBit ? stride0 : stride1,
      };
    };


#ifdef __cpp_concepts
    template<OpenKalman::Eigen3::eigen_general T>
    struct EigenWrapperTraits<T>
#else
    template<typename T>
    struct EigenWrapperTraits<T, std::enable_if_t<OpenKalman::Eigen3::eigen_general<T>>>
#endif
      : traits<T>
    {
      using ElementRef = decltype(OpenKalman::get_element(std::declval<std::remove_reference_t<T>&>(), 0, 0));
      static constexpr bool lvb = std::is_same_v<ElementRef, std::decay_t<ElementRef>&>;
      static constexpr bool nest_is_big = not std::is_lvalue_reference_v<T> and
        ((traits<T>::Flags & NestByRefBit) != 0x0 or std::is_lvalue_reference_v<typename std::decay_t<T>::Nested>);
      enum {
        Flags = (traits<T>::Flags & ~NestByRefBit & ~LvalueBit) | (nest_is_big ? NestByRefBit : 0x0) | (lvb ? LvalueBit : 0x0),
      };
    };

  } // namespace OpenKalman_detail


  template<typename T, typename L>
  struct traits<OpenKalman::internal::LibraryWrapper<T, L>> : OpenKalman_detail::EigenWrapperTraits<std::decay_t<T>>
  {
    static_assert(OpenKalman::Eigen3::eigen_general<L>);
    using Scalar = OpenKalman::scalar_type_of_t<T>;
    using XprKind = std::conditional_t<OpenKalman::Eigen3::eigen_array_general<T, true>, ArrayXpr, MatrixXpr>;
    using StorageIndex = Index;
    using StorageKind = Dense;
    enum {
      RowsAtCompileTime = OpenKalman::dynamic_dimension<T, 0> ? Eigen::Dynamic : static_cast<Index>(OpenKalman::index_dimension_of_v<T, 0>),
      ColsAtCompileTime = OpenKalman::dynamic_dimension<T, 1> ? Eigen::Dynamic : static_cast<Index>(OpenKalman::index_dimension_of_v<T, 1>),
      MaxRowsAtCompileTime [[maybe_unused]] = RowsAtCompileTime,
      MaxColsAtCompileTime [[maybe_unused]] = ColsAtCompileTime,
    };
  };


  // ---------------------------- //
  //  Eigen::internal::evaluator  //
  // ---------------------------- //

  namespace OpenKalman_detail
  {
#ifdef __cpp_concepts
    template<typename XprType>
#else
    template<typename XprType, typename = void>
#endif
    struct EigenWrapperEvaluator : evaluator_base<OpenKalman::Eigen3::EigenWrapper<XprType>>
    {
    private:

      using ElementRef = decltype(OpenKalman::get_element(std::declval<XprType&>(), 0, 0));

    public:

      explicit EigenWrapperEvaluator(const XprType& t) : m_xpr {const_cast<XprType&>(t)} {}

#ifdef __cpp_concepts
      auto& coeffRef(Index row, Index col) requires std::same_as<ElementRef, std::decay_t<ElementRef>&>
#else
      template<bool nc = std::is_same_v<ElementRef, std::decay_t<ElementRef>&>, std::enable_if_t<nc, int> = 0>
      auto& coeffRef(Index row, Index col)
#endif
      {
        return OpenKalman::get_element(m_xpr, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }


      constexpr decltype(auto) coeff(Index row, Index col) const
      {
        return OpenKalman::get_element(m_xpr, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }

      enum {
        CoeffReadCost = 0,
        Flags = traits<OpenKalman::Eigen3::EigenWrapper<XprType>>::Flags,
        Alignment = AlignedMax
      };

    protected:

      XprType& m_xpr;
    };


#ifdef __cpp_concepts
    template<OpenKalman::Eigen3::eigen_dense_general<true> ArgType>
    struct EigenWrapperEvaluator<ArgType>
#else
    template<typename ArgType>
    struct EigenWrapperEvaluator<ArgType, std::enable_if_t<OpenKalman::Eigen3::eigen_dense_general<ArgType, true>>>
#endif
      : evaluator<std::decay_t<ArgType>>
    {
      using XprType = std::decay_t<ArgType>;
      explicit EigenWrapperEvaluator(const XprType& t) : evaluator<XprType> {t} {}
    };

  } // namespace OpenKalman_detail


  template<typename T, typename L>
  struct evaluator<OpenKalman::internal::LibraryWrapper<T, L>> : OpenKalman_detail::EigenWrapperEvaluator<std::remove_reference_t<T>>
  {
    static_assert(OpenKalman::Eigen3::eigen_general<L>);
    using XprType = OpenKalman::Eigen3::EigenWrapper<T>;
    using Base = OpenKalman_detail::EigenWrapperEvaluator<std::remove_reference_t<T>>;
    explicit evaluator(const XprType& t) : Base {t.nested_matrix()} {}
  };

} // Eigen::internal

#endif //OPENKALMAN_EIGENWRAPPER_HPP