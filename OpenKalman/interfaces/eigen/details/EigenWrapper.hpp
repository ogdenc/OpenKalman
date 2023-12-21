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
 * \brief Definitions for Eigen3::EigenWrapper (an alias for LibraryWrapper in the Eigen library)
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
    struct EigenWrapperTraitsBase
    {
    private:

      using ElementRef = decltype(OpenKalman::get_component(std::declval<std::add_lvalue_reference_t<T>>(), 0, 0));
      static constexpr auto lvalue_bit = std::is_same_v<ElementRef, std::decay_t<ElementRef>&> ? LvalueBit : 0x0;
      static constexpr auto layout_bit = OpenKalman::layout_of_v<T> == OpenKalman::Layout::right ? RowMajorBit : 0x0;
      static constexpr auto direct = OpenKalman::directly_accessible<T> ? DirectAccessBit : 0x0;

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

      using IndexType = typename std::decay_t<T>::Index;

      static constexpr IndexType stride0 = OpenKalman::static_index_value<Stride0, std::ptrdiff_t> ? static_cast<std::ptrdiff_t>(Stride0{}) : Eigen::Dynamic;
      static constexpr IndexType stride1 = OpenKalman::static_index_value<Stride1, std::ptrdiff_t> ? static_cast<std::ptrdiff_t>(Stride1{}) : Eigen::Dynamic;

    public:

      using XprKind = MatrixXpr;
      enum {
        Flags = layout_bit | direct | lvalue_bit,
        InnerStrideAtCompileTime = layout_bit == RowMajorBit ? stride1 : stride0,
        OuterStrideAtCompileTime = layout_bit == RowMajorBit ? stride0 : stride1,
      };
    };


#ifdef __cpp_concepts
    template<OpenKalman::Eigen3::eigen_general T>
    struct EigenWrapperTraitsBase<T>
#else
    template<typename T>
    struct EigenWrapperTraitsBase<T, std::enable_if_t<OpenKalman::Eigen3::eigen_general<T>>>
#endif
      : traits<std::decay_t<T>>
    {
    private:

      using NestedTraits = traits<std::decay_t<T>>;
      using GetRes = decltype(OpenKalman::get_component(std::declval<std::add_lvalue_reference_t<T>>(),
        std::declval<std::size_t>(), std::declval<std::size_t>()));

      static constexpr auto lvalue_bit = (NestedTraits::Flags & LvalueBit) != 0x0 and (OpenKalman::Eigen3::eigen_dense_general<T, true> or
          (std::is_lvalue_reference_v<GetRes> and not std::is_const_v<std::remove_reference_t<GetRes>>)) ? LvalueBit : 0x0;

    public:

      enum {
        Flags = (NestedTraits::Flags & ~LvalueBit) | lvalue_bit,
      };
    };

  } // namespace OpenKalman_detail


  template<typename T, typename L, typename...Ps>
  struct traits<OpenKalman::internal::LibraryWrapper<T, L, Ps...>> : OpenKalman_detail::EigenWrapperTraitsBase<T>
  {
  private:

    using Base = OpenKalman_detail::EigenWrapperTraitsBase<T>;
    static constexpr auto nest_by_ref = sizeof...(Ps) > 0 and not std::is_lvalue_reference_v<T> ? NestByRefBit : 0x0;

  public:

    static_assert(OpenKalman::Eigen3::eigen_general<L>);
    using Scalar = OpenKalman::scalar_type_of_t<T>;
    using XprKind = std::conditional_t<OpenKalman::Eigen3::eigen_array_general<T, true>, ArrayXpr, MatrixXpr>;
    using StorageIndex = Index;
    using StorageKind = Dense;
    enum {
      Flags = (Base::Flags & ~NestByRefBit) | nest_by_ref,
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
    template<typename XprType, typename Nested>
#else
    template<typename XprType, typename Nested, typename = void>
#endif
    struct EigenWrapperEvaluatorBase : evaluator_base<XprType>
    {
      explicit EigenWrapperEvaluatorBase(const XprType& t) : m_xpr {const_cast<XprType&>(t)} {}


      auto& coeffRef(Index row, Index col)
      {
        return OpenKalman::get_component(m_xpr.nested_object(), static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }


      constexpr decltype(auto) coeff(Index row, Index col) const
      {
        return OpenKalman::get_component(m_xpr.nested_object(), static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }


      enum {
        CoeffReadCost = 0,
        Flags = traits<XprType>::Flags,
        Alignment = AlignedMax
      };

    protected:

      XprType& m_xpr;
    };


#ifdef __cpp_concepts
    template<typename XprType, OpenKalman::Eigen3::eigen_dense_general Nested>
    struct EigenWrapperEvaluatorBase<XprType, Nested>
#else
    template<typename XprType, typename Nested>
    struct EigenWrapperEvaluatorBase<XprType, Nested, std::enable_if_t<OpenKalman::Eigen3::eigen_dense_general<Nested>>>
#endif
      : evaluator<std::decay_t<Nested>>
    {
      explicit EigenWrapperEvaluatorBase(const XprType& t) : evaluator<std::decay_t<Nested>> {t.nested_object()} {}
    };

  } // namespace OpenKalman_detail


  template<typename T, typename L, typename...Ps>
  struct evaluator<OpenKalman::internal::LibraryWrapper<T, L, Ps...>>
    : OpenKalman_detail::EigenWrapperEvaluatorBase<OpenKalman::internal::LibraryWrapper<T, L, Ps...>, T>
  {
    static_assert(OpenKalman::Eigen3::eigen_general<L>);
    using XprType = OpenKalman::internal::LibraryWrapper<T, L, Ps...>;
    using Base = OpenKalman_detail::EigenWrapperEvaluatorBase<XprType, T>;
    explicit evaluator(const XprType& t) : Base {t} {}
  };

} // Eigen::internal

#endif //OPENKALMAN_EIGENWRAPPER_HPP